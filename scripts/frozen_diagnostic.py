"""Synthetic CPU diagnostic ONLY: no data download, fitting, or optimizer step."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformer_gsnr.control import calibrate, make_optimizer, MomentumMap, update_momenta
from transformer_gsnr.model import SmallViT, ViTConfig
from transformer_gsnr.task import DistillationTask, parameter_layout


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    architecture = dict(image_size=8, patch_size=4, classes=6, depth=2, heads=3, mlp_ratio=2)
    student = SmallViT(ViTConfig(**architecture, width=12))
    teacher = SmallViT(ViTConfig(**architecture, width=18)).eval().requires_grad_(False)
    task = DistillationTask(student, mode="feature", teacher_width=18)
    generator = torch.Generator().manual_seed(13)
    batches = [(torch.randn(8, 3, 8, 8, generator=generator), torch.randint(6, (8,), generator=generator)) for _ in range(8)]
    result = calibrate(task, batches, teacher)
    optimizer = make_optimizer(task, lr=.01)
    result["mapped_momenta"] = update_momenta(optimizer, result, MomentumMap())
    result["scope"] = "synthetic random images, labels, student and teacher; correctness diagnostic only"
    result["training_steps"] = 0
    result["student_parameters"] = sum(p.numel() for p in student.parameters())
    result["parameter_mapping"] = [{"parameter": name, "layer": layer, "size": p.numel()}
        for (name, p), layer in zip(*parameter_layout(task))]
    result["torch"] = torch.__version__
    destination = Path(__file__).resolve().parents[1] / "reports/frozen-diagnostic.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(destination), "isolation": result["isolation"]["unchanged"],
                      "cells": len(result["rows"]), "training_steps": 0}))


if __name__ == "__main__":
    main()
