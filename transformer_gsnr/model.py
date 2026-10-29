"""Small pre-LayerNorm ViT for controlled experiments, not a DeiT replica."""
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    classes: int = 100
    width: int = 192
    depth: int = 6
    heads: int = 3
    mlp_ratio: int = 4
    dropout: float = 0.

    def __post_init__(self):
        for name in ("image_size", "patch_size", "classes", "width", "depth", "heads", "mlp_ratio"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size % self.patch_size or self.width % self.heads or self.classes < 2:
            raise ValueError("Patch/head divisibility or class count invalid")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.width
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * config.mlp_ratio), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(d * config.mlp_ratio, d),
                                 nn.Dropout(config.dropout))

    def forward(self, x):
        q = self.norm1(x)
        # Explicit attention-weight path supports deterministic backward on CPU/CUDA.
        attended, _ = self.attn(q, q, q, need_weights=True)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class SmallViT(nn.Module):
    def __init__(self, config=ViTConfig()):
        super().__init__()
        self.config = config
        d = config.width
        self.patch_embed = nn.Conv2d(3, d, config.patch_size, config.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, (config.image_size // config.patch_size)**2 + 1, d))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.classes)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, self.config.image_size, self.config.image_size):
            raise ValueError("Image shape differs from ViTConfig")
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(len(images), -1, -1), tokens), dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        feature = self.norm(x)[:, 0]
        return self.head(feature), feature

    def architecture(self):
        return asdict(self.config)
