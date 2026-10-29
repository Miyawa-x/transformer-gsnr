"""Finite-window sufficient statistics and isolated diagnostic forwards."""
from contextlib import contextmanager
import copy
import hashlib
import random

import numpy as np
import torch


def state_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous()
            digest.update(str((array.dtype, tuple(array.shape))).encode())
            digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


@contextmanager
def isolated_probe(models, optimizer=None, *, training=True):
    """Restore tensors, modes, existing .grad, optimizer and RNG even on error."""
    models = tuple(models)
    model_states = [copy.deepcopy(model.state_dict()) for model in models]
    modes = [[(module, module.training) for module in model.modules()] for model in models]
    gradients = [[None if p.grad is None else p.grad.clone() for p in m.parameters()] for m in models]
    optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    random_state = rng_state()
    audit = {"before": state_hash([model_states, gradients, optimizer_state, random_state,
                                  [[v for _, v in rows] for rows in modes]])}
    # The caller passes only the student here when the teacher must stay eval.
    for model in models:
        model.train(training)
    try:
        yield audit
    finally:
        for model, saved, saved_grads, saved_modes in zip(models, model_states, gradients, modes):
            model.load_state_dict(saved)
            for parameter, gradient in zip(model.parameters(), saved_grads):
                parameter.grad = gradient
            for module, mode in saved_modes:
                module.training = mode
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
        restore_rng(random_state)
        audit["after"] = state_hash([[model.state_dict() for model in models],
                                     [[p.grad for p in m.parameters()] for m in models],
                                     optimizer.state_dict() if optimizer is not None else None, rng_state(),
                                     [[module.training for module in model.modules()] for model in models]])
        audit["unchanged"] = audit["before"] == audit["after"]
        if not audit["unchanged"]:
            raise RuntimeError("Probe changed training state")


class GradientWindow:
    """One fixed-dimensional gradient stream, grouped by parameter owner layer."""
    def __init__(self, named_parameters, epsilon=1e-12):
        self.names, self.parameters = zip(*named_parameters)
        self.sums = [torch.zeros_like(p, device="cpu", dtype=torch.float64) for p in self.parameters]
        self.square_sums = torch.zeros(len(self.parameters), dtype=torch.float64)
        self.count = 0
        self.epsilon = epsilon

    @torch.no_grad()
    def observe(self, gradients):
        if len(gradients) != len(self.parameters):
            raise ValueError("Gradient count mismatch")
        for i, (gradient, parameter) in enumerate(zip(gradients, self.parameters)):
            if gradient is None:
                continue  # Unused coordinates are zeros, not missing observations.
            if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
                raise ValueError("Nonfinite or incorrectly shaped gradient")
            value = gradient.detach().to(device="cpu", dtype=torch.float64)
            self.sums[i].add_(value)
            self.square_sums[i] += value.square().sum()
        self.count += 1

    def sufficient_statistics(self):
        return {"names": self.names, "gradient_sums": self.sums,
                "squared_norm_sums": self.square_sums, "observations": self.count}

    def rows(self):
        if self.count < 2:
            raise ValueError("GSNR requires at least two observations")
        cells = {}
        for i, name in enumerate(self.names):
            layer = name.rsplit(".", 1)[0] if "." in name else name
            mean_power = float(self.sums[i].square().sum()) / self.count**2
            second = float(self.square_sums[i]) / self.count
            for key in (layer, "__global__"):
                cell = cells.setdefault(key, [0.0, 0.0])
                cell[0] += mean_power
                cell[1] += second
        result = []
        for layer, (mean_power, second) in cells.items():
            noise_raw = (second - mean_power) * self.count / (self.count - 1)
            noise = max(noise_raw, 0.0)
            signal_raw = mean_power - noise / self.count
            result.append({"layer": layer, "observations": self.count,
                           "mean_gradient_norm_squared": mean_power, "second_moment": second,
                           "noise_trace_unclipped": noise_raw, "noise_trace": noise,
                           "signal_unclipped": signal_raw, "signal_clipped": max(signal_raw, 0.0),
                           "epsilon": self.epsilon,
                           "gsnr": max(signal_raw, 0.0) / (noise + self.epsilon)})
        return result
