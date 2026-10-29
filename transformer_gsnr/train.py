"""CIFAR-100 server-only training entry; CPU is reserved for correctness tests."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .control import MomentumMap, calibrate, diagnostic_rates, make_optimizer, update_momenta
from .model import SmallViT, ViTConfig
from .task import DistillationTask, gradients
from .vendor.measurement import isolated_probe, restore_rng, rng_state


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def split_indices(targets, seed):
    """40k train / 5k validation / 5k calibration; official test is untouched."""
    generator = torch.Generator().manual_seed(seed)
    indices = {k: [] for k in ("train", "validation", "calibration")}
    targets = torch.tensor(targets)
    for label in range(100):
        values = torch.where(targets == label)[0]
        if len(values) != 500:
            raise ValueError("Expected CIFAR-100 training counts")
        values = values[torch.randperm(len(values), generator=generator)].tolist()
        indices["calibration"] += values[:50]
        indices["validation"] += values[50:100]
        indices["train"] += values[100:]
    return indices


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    correct = correct5 = count = 0
    with isolated_probe([model], training=False):
        for images, targets in loader:
            logits, _ = model(images.to(device))
            targets = targets.to(device)
            ranks = logits.topk(5, dim=1).indices
            correct += int(ranks[:, 0].eq(targets).sum())
            correct5 += int(ranks.eq(targets[:, None]).any(1).sum())
            count += len(targets)
    if not count:
        raise ValueError("Empty evaluation set")
    return {"top1": correct / count, "top5": correct5 / count, "samples": count}


def validate_config(config):
    for name in ("epochs", "batch_size", "probe_batches", "probe_batch_size", "calibrate_every"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError("Invalid " + name)
    if config["probe_batches"] < 2 or config["probe_batches"] * config["probe_batch_size"] > 5000:
        raise ValueError("Probe must fit calibration split and have at least two batches")
    if config["optimizer"] not in ("adaptive", "fixed", "adamw"):
        raise ValueError("Invalid optimizer")
    if not 0 < config["lr"] < float("inf") or config["weight_decay"] != 0:
        raise ValueError("Prototype requires finite positive LR and zero decay for clean optimizer controls")
    if not 0 <= config["warmup_epochs"] < config["epochs"] or config["workers"] < 0:
        raise ValueError("Invalid warmup/workers")
    MomentumMap(**config["momentum_map"])
    if not 0 <= config["momentum"] < 1:
        raise ValueError("Invalid fixed momentum")
    if config["objective"] not in ("ce", "kd", "feature"):
        raise ValueError("Invalid objective")
    architecture = ViTConfig(**config["model"])
    if architecture.image_size != 32 or architecture.classes != 100:
        raise ValueError("CIFAR runner requires image32 and 100 classes")


def run(args):
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Run dataset training on a CUDA server; local CPU is for unit tests only")
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    if config["objective"] != "ce" and (not args.teacher or not args.teacher_sha256 or args.expected_teacher_top1 is None):
        raise ValueError("KD needs a trusted teacher export, SHA256, and expected validation Top1")
    if config["objective"] != "ce" and (not math.isfinite(args.expected_teacher_top1) or not 0 <= args.expected_teacher_top1 <= 1):
        raise ValueError("Expected teacher Top1 must be a finite fraction in [0,1]")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    # Prevent two processes from resuming into one destination.
    import fcntl
    lock = (output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args._owns_output = True
    atomic_json(output / "status.json", {"state": "preparing", "resume": args.resume})
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = args.device
    norm = transforms.Normalize((.5071, .4867, .4408), (.2675, .2565, .2761))
    eval_transform = transforms.Compose([transforms.ToTensor(), norm])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    clean = datasets.CIFAR100(args.data, train=True, download=args.download, transform=eval_transform)
    augmented = datasets.CIFAR100(args.data, train=True, download=False, transform=train_transform)
    split = split_indices(clean.targets, config["split_seed"])
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    rng = torch.Generator().manual_seed(config["probe_seed"])
    # Independent draws from the empirical calibration distribution; replacement
    # matches the iid microbatch estimator rather than a disjoint finite-population correction.
    draws = torch.randint(len(split["calibration"]),
        (config["probe_batches"] * config["probe_batch_size"],), generator=rng).tolist()
    probe_ids = [split["calibration"][i] for i in draws]
    probe_loader = DataLoader(Subset(clean, probe_ids), batch_size=config["probe_batch_size"], num_workers=0,
                              generator=torch.Generator().manual_seed(config["probe_seed"]))
    # Fixed CPU tensors prevent repeated augmentation or loader RNG from changing probes.
    probe_batches = list(probe_loader)
    validation = DataLoader(Subset(clean, split["validation"]), batch_size=256, num_workers=0,
                            generator=torch.Generator().manual_seed(config["split_seed"]))
    teacher = None
    teacher_score = None
    if config["objective"] != "ce":
        if sha256(args.teacher) != args.teacher_sha256:
            raise ValueError("Teacher checksum mismatch")
        saved_teacher = torch.load(args.teacher, map_location="cpu", weights_only=True)
        if saved_teacher["dataset"] != "cifar100" or saved_teacher["class_names"] != clean.classes:
            raise ValueError("Teacher dataset/class ordering mismatch")
        if saved_teacher.get("split_sha256") != split_hash:
            raise ValueError("Teacher must use the same held-out validation split")
        teacher = SmallViT(ViTConfig(**saved_teacher["architecture"])).to(device)
        if teacher.config.image_size != 32:
            raise ValueError("Teacher input size differs")
        teacher.load_state_dict(saved_teacher["model"], strict=True)
        teacher.requires_grad_(False).eval()
        teacher_score = evaluate(teacher, validation, device)
        if abs(teacher_score["top1"] - args.expected_teacher_top1) > .002:
            raise ValueError("Teacher validation accuracy differs by >0.2 percentage points")
    # Teacher construction/loading must not change the student's initialization.
    torch.manual_seed(config["seed"])
    student = SmallViT(ViTConfig(**config["model"])).to(device)
    task = DistillationTask(student, mode=config["objective"], temperature=config["temperature"],
        teacher_width=teacher.config.width if teacher else None).to(device)
    optimizer = (torch.optim.AdamW(task.parameters(), lr=config["lr"], weight_decay=0.)
        if config["optimizer"] == "adamw" else make_optimizer(task, lr=config["lr"], momentum=config["momentum"]))
    mapping = MomentumMap(**config["momentum_map"])
    source_root = Path(__file__).parent
    identity = {"config": config, "split_sha256": split_hash, "probe_indices": probe_ids,
        "teacher_sha256": args.teacher_sha256, "teacher_validation": teacher_score,
        "class_names": clean.classes,
        "sources": {str(p.relative_to(source_root)): sha256(p) for p in sorted(source_root.rglob("*.py"))},
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)}
    history, start, best, previous, previous_epoch = [], 0, -1., None, None
    if args.resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume identity differs (config/data/teacher/code/environment)")
        task.load_state_dict(saved["task"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history, start, best = saved["history"], saved["epoch"], saved["best"]
        previous, previous_epoch = saved["previous_measurement"], saved["previous_epoch"]
    else:
        atomic_json(output / "manifest.json", identity)
        atomic_json(output / "split.json", split)

    def measure(epoch):
        nonlocal previous, previous_epoch
        result = calibrate(task, probe_batches, teacher, device=device)
        result["epoch"] = epoch
        result["rates"] = diagnostic_rates(result, previous, epoch - previous_epoch) if previous is not None else []
        result["momentum_updates"] = update_momenta(optimizer, result, mapping) if config["optimizer"] == "adaptive" else []
        atomic_json(output / f"gsnr_{epoch:03d}.json", result)
        previous, previous_epoch = result, epoch

    if not args.resume:
        measure(0)
    started = time.monotonic()
    for epoch in range(start + 1, config["epochs"] + 1):
        warm = min(1., epoch / max(1, config["warmup_epochs"]))
        phase = max(0, epoch - config["warmup_epochs"]) / (config["epochs"] - config["warmup_epochs"])
        lr = config["lr"] * warm * .5 * (1. + np.cos(np.pi * phase))
        for group in optimizer.param_groups:
            group["lr"] = float(lr)
        loader = DataLoader(Subset(augmented, split["train"]), batch_size=config["batch_size"], shuffle=True,
            num_workers=config["workers"], worker_init_fn=seed_worker, pin_memory=True,
            generator=torch.Generator().manual_seed(config["seed"] + epoch * 1000003))
        task.train()
        sums, seen = dict.fromkeys(task.streams, 0.), 0
        for step, (images, targets) in enumerate(loader):
            losses = task(images.to(device), targets.to(device), teacher)
            if config["optimizer"] == "adamw":
                optimizer.zero_grad(set_to_none=True)
                sum(losses.values()).backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in task.parameters()):
                    raise FloatingPointError("Nonfinite gradients")
                optimizer.step()
            else:
                gs = gradients(losses, optimizer.ordered_parameters())
                if any(g is not None and not torch.isfinite(g).all() for values in gs.values() for g in values):
                    raise FloatingPointError("Nonfinite gradients")
                optimizer.step(gs)
            seen += len(targets)
            for s, loss in losses.items():
                sums[s] += float(loss.detach()) * len(targets)
            if step % 50 == 0:
                progress = {"state": "training", "epoch": epoch, "step": step, "samples": seen}
                atomic_json(output / "status.json", progress)
                print(json.dumps(progress), flush=True)
        scores = evaluate(student, validation, device)
        if epoch % config["calibrate_every"] == 0 or epoch == config["epochs"]:
            measure(epoch)
        row = {"epoch": epoch, "lr": float(lr), "losses": {s: v / seen for s, v in sums.items()},
               "validation": scores, "elapsed_this_invocation": time.monotonic() - started}
        history.append(row)
        if scores["top1"] > best:
            best = scores["top1"]
            atomic_save(output / "teacher_export.pt", {"model": student.state_dict(), "architecture": student.architecture(),
                "dataset": "cifar100", "class_names": clean.classes, "validation": scores,
                "split_sha256": split_hash, "epoch": epoch})
        atomic_save(output / "last.pt", {"identity": identity, "task": task.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": rng_state(), "epoch": epoch, "history": history,
            "best": best, "previous_measurement": previous, "previous_epoch": previous_epoch})
        atomic_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
    atomic_json(output / "status.json", {"state": "completed", "epochs": config["epochs"],
        "best_validation_top1": best, "official_test_evaluated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--teacher-sha256")
    parser.add_argument("--expected-teacher-top1", type=float, help="fraction on the SAME validation split")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        failure = {"state": "failed", "error": repr(error)}
        if getattr(args, "_owns_output", False):
            atomic_json(Path(args.output) / "status.json", failure)
        print(json.dumps(failure), flush=True)
        raise


if __name__ == "__main__":
    main()
