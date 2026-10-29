"""Frozen-parameter calibration; GSNR is not the training loss."""
from dataclasses import dataclass
import math

import torch

from .task import gradients, parameter_layout
from .vendor.measurement import GradientWindow, isolated_probe
from .vendor.optimizer import LayerwiseMultiStreamSGD


@dataclass(frozen=True)
class MomentumMap:
    minimum: float = .5
    maximum: float = .95
    half_saturation: float = 1.

    def __post_init__(self):
        if not (0 <= self.minimum < self.maximum < 1) or not math.isfinite(self.half_saturation) or self.half_saturation <= 0:
            raise ValueError("Invalid monotone momentum map")

    def __call__(self, gsnr):
        if not math.isfinite(gsnr) or gsnr < 0:
            raise ValueError("GSNR must be finite and nonnegative")
        fraction = 0. if gsnr == 0 else 1. / (1. + self.half_saturation / gsnr)
        return self.minimum + (self.maximum - self.minimum) * fraction


def make_optimizer(task, *, lr, momentum=.9, weight_decay=0.):
    named, layers = parameter_layout(task)
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Invalid learning rate or weight decay")
    return LayerwiseMultiStreamSGD([p for _, p in named], layers, lr=lr,
        initial_momentums={s: momentum for s in task.streams}, weight_decay=weight_decay, decay_stream="tog")


def calibrate(task, batches, teacher=None, *, device="cpu"):
    """Independent equal-size microbatches at fixed theta; no update performed.

    eval-mode student disables dropout: these are data-noise statistics, not
    noise from stochastic dropout. Statistics exclude L2 and clipping.
    """
    named, layers = parameter_layout(task)
    parameters = tuple(p for _, p in named)
    grouped = tuple((layer + "." + str(i), p) for i, (layer, p) in enumerate(zip(layers, parameters)))
    windows = {s: GradientWindow(grouped) for s in task.streams}
    activity = {(s, layer): 0 for s in task.streams for layer in set(layers)}
    batch_size = None
    models = [task] + ([teacher] if teacher is not None else [])
    with isolated_probe(models, training=False) as isolation:
        for images, targets in batches:
            if batch_size is None:
                batch_size = len(targets)
            if batch_size < 1 or len(targets) != batch_size:
                raise ValueError("Calibration microbatches must have equal nonzero size")
            losses = task(images.to(device), targets.to(device), teacher)
            gs = gradients(losses, parameters)
            for stream, window in windows.items():
                window.observe(gs[stream])
                used = {layer for layer, gradient in zip(layers, gs[stream]) if gradient is not None}
                for layer in used:
                    activity[stream, layer] += 1
    rows = []
    for stream, window in windows.items():
        for row in window.rows():
            active = (window.count if row["layer"] == "__global__"
                      else activity[stream, row["layer"]])
            rows.append({**row, "stream": stream, "active_observations": active,
                         "microbatch_size": batch_size, "scope": "frozen_eval_data_noise"})
    return {"rows": rows, "isolation": isolation}


def update_momenta(optimizer, measurement, mapping):
    """Keep history intact: update coefficients only, never clear buffers."""
    table = optimizer.layer_momentums
    updates = []
    for row in measurement["rows"]:
        if row["layer"] == "__global__":
            continue
        layer, stream = row["layer"], row["stream"]
        if row["active_observations"] != row["observations"]:
            # Feature loss never touches the classifier; no fake noise estimate.
            if row["active_observations"] != 0:
                raise ValueError("Intermittently active cells require a separate missingness model")
            continue
        old = table[layer][stream]
        table[layer][stream] = mapping(row["gsnr"])
        updates.append({"layer": layer, "stream": stream, "old_momentum": old,
                        "momentum": table[layer][stream], "gsnr": row["gsnr"],
                        "steady_gain_proxy": 1. / (1. - table[layer][stream])})
    optimizer.set_layer_momentums(table)
    return updates


def diagnostic_rates(current, previous, epoch_delta):
    if epoch_delta <= 0:
        raise ValueError("Positive measurement interval required")
    old = {(r["layer"], r["stream"]): r for r in previous["rows"]}
    result = []
    for row in current["rows"]:
        a = old[row["layer"], row["stream"]]["gsnr"]
        b = row["gsnr"]
        result.append({"layer": row["layer"], "stream": row["stream"],
                       "gsnr": b, "absolute_drop_per_epoch": (a - b) / epoch_delta,
                       "log_decay_per_epoch": math.log(a / b) / epoch_delta if min(a, b) > 0 else None})
    return result
