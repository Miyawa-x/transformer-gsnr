"""Original TOG/TCG/NCG semantics, optional explicit CLS feature adapter."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.losses import dkd_component_losses


class DistillationTask(nn.Module):
    def __init__(self, student, *, mode="ce", temperature=4., teacher_width=None,
                 alpha=1., beta=1., feature_weight=1.):
        super().__init__()
        if mode not in ("ce", "kd", "feature"):
            raise ValueError("Unknown objective")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid temperature")
        if any(not math.isfinite(w) or w <= 0 for w in (alpha, beta, feature_weight)):
            raise ValueError("Loss weights must be finite and positive")
        if mode == "feature" and (teacher_width is None or teacher_width < 1):
            raise ValueError("Feature KD needs the teacher feature width")
        self.student, self.mode, self.temperature = student, mode, temperature
        self.alpha, self.beta, self.feature_weight = alpha, beta, feature_weight
        self.feature_adapter = nn.Linear(student.config.width, teacher_width) if mode == "feature" else None
        self.streams = ("tog",) if mode == "ce" else ("tog", "tcg", "ncg")
        if mode == "feature":
            self.streams += ("feature",)

    def forward(self, images, targets, teacher=None):
        logits, features = self.student(images)
        losses = {"tog": F.cross_entropy(logits, targets)}
        if self.mode != "ce":
            if teacher is None or teacher.training:
                raise ValueError("KD requires an eval-mode teacher")
            with torch.no_grad():
                teacher_logits, teacher_features = teacher(images)
            if teacher_logits.shape != logits.shape:
                raise ValueError("Teacher and student must use the same ordered classes")
            tcg, ncg = dkd_component_losses(logits, teacher_logits, targets, self.temperature)
            losses.update(tcg=self.alpha * tcg, ncg=self.beta * ncg)
            if self.mode == "feature":
                projected = F.normalize(self.feature_adapter(features), dim=-1)
                target_features = F.normalize(teacher_features.detach(), dim=-1)
                losses["feature"] = self.feature_weight * F.mse_loss(projected, target_features)
        if not all(torch.isfinite(v).all() for v in losses.values()):
            raise FloatingPointError("Nonfinite loss")
        return losses


def parameter_layout(task):
    """Disjoint attention/MLP/LayerNorm groups, explicit embedding/head/adapter."""
    named = tuple(task.named_parameters())
    layers = []
    for name, parameter in named:
        if not parameter.requires_grad:
            raise ValueError("This prototype expects all student/adapter parameters trainable")
        if name.startswith("feature_adapter."):
            layer = "feature_adapter"
        elif name in ("student.cls_token", "student.pos_embed") or name.startswith("student.patch_embed."):
            layer = "embedding"
        elif name.startswith("student.blocks."):
            parts = name.split(".")
            if parts[3] not in ("attn", "mlp", "norm1", "norm2"):
                raise ValueError("Unknown Transformer submodule: " + name)
            layer = ".".join(parts[1:4])
        elif name.startswith("student.norm."):
            layer = "final_norm"
        elif name.startswith("student.head."):
            layer = "head"
        else:
            raise ValueError("Unmapped parameter: " + name)
        layers.append(layer)
    return named, tuple(layers)


def gradients(losses, parameters):
    parameters = tuple(parameters)
    return {stream: torch.autograd.grad(loss, parameters, retain_graph=i < len(losses) - 1,
                                        allow_unused=True)
            for i, (stream, loss) in enumerate(losses.items())}
