"""Named-stream Heavy-Ball SGD with frozen layer-wise control tables."""

import math
from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from .streams import (
    LEGACY_STREAM_NAMES,
    require_exact_streams,
    validate_stream_names,
)


CHECKPOINT_SCHEMA_VERSION = 3
LEGACY_LAYERWISE_SCHEMA_VERSION = 2


class LayerwiseMultiStreamSGD(Optimizer):
    """Apply momentum and injection weight per ``(layer, stream)`` cell.

    Changing a coefficient never clears or rescales an existing buffer. This
    preserves the unnormalised Heavy-Ball recurrence across calibration
    boundaries: ``v_t = mu_t * v_(t-1) + g_t``.
    """

    def __init__(
        self,
        parameters: Sequence[Tensor],
        layer_names: Sequence[str],
        *,
        lr: float,
        initial_momentums: Mapping[str, float],
        weight_decay: float = 0.0,
        decay_stream: Optional[str] = None,
    ) -> None:
        parameters = tuple(parameters)
        self.layer_names = tuple(layer_names)
        self.stream_names = validate_stream_names(initial_momentums)
        if len(parameters) != len(self.layer_names):
            raise ValueError("parameters and layer_names must have equal length")
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if decay_stream is None:
            decay_stream = "tog" if "tog" in self.stream_names else "outside"
        decay_placements = (*self.stream_names, "outside")
        if decay_stream not in decay_placements:
            raise ValueError(f"decay_stream must be one of {decay_placements}")
        checked = {}
        for stream in self.stream_names:
            momentum = float(initial_momentums[stream])
            if not 0.0 <= momentum < 1.0:
                raise ValueError(f"Invalid momentum for {stream}: {momentum}")
            checked[stream] = momentum

        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "decay_stream": decay_stream,
        }
        super().__init__(parameters, defaults)
        self._layer_momentums = {
            layer: dict(checked) for layer in dict.fromkeys(self.layer_names)
        }
        self._layer_stream_weights = {
            layer: {stream: 1.0 for stream in self.stream_names}
            for layer in dict.fromkeys(self.layer_names)
        }

    def ordered_parameters(self) -> tuple[Tensor, ...]:
        return tuple(parameter for group in self.param_groups for parameter in group["params"])

    @property
    def layer_momentums(self) -> dict[str, dict[str, float]]:
        return {layer: dict(values) for layer, values in self._layer_momentums.items()}

    @property
    def layer_stream_weights(self) -> dict[str, dict[str, float]]:
        return {
            layer: dict(values)
            for layer, values in self._layer_stream_weights.items()
        }

    def set_layer_momentums(
        self,
        momentums: Mapping[str, Mapping[str, float]],
        *,
        strict: bool = True,
    ) -> None:
        if strict and set(momentums) != set(self._layer_momentums):
            raise ValueError("Momentum table does not match optimizer layers")
        updated = {layer: dict(values) for layer, values in self._layer_momentums.items()}
        for layer, stream_values in momentums.items():
            if layer not in updated:
                raise ValueError(f"Unknown layer: {layer}")
            require_exact_streams(
                stream_values,
                self.stream_names,
                field=f"momentums[{layer!r}]",
            )
            for stream in self.stream_names:
                value = float(stream_values[stream])
                if not 0.0 <= value < 1.0:
                    raise ValueError(f"Invalid momentum for {layer}/{stream}: {value}")
                updated[layer][stream] = value
        self._layer_momentums = updated

    def set_layer_stream_weights(
        self,
        weights: Mapping[str, Mapping[str, float]],
        *,
        strict: bool = True,
    ) -> None:
        if strict and set(weights) != set(self._layer_stream_weights):
            raise ValueError("Stream-weight table does not match optimizer layers")
        updated = {
            layer: dict(values)
            for layer, values in self._layer_stream_weights.items()
        }
        for layer, stream_values in weights.items():
            if layer not in updated:
                raise ValueError(f"Unknown layer: {layer}")
            require_exact_streams(
                stream_values,
                self.stream_names,
                field=f"stream_weights[{layer!r}]",
            )
            for stream in self.stream_names:
                value = float(stream_values[stream])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"Invalid stream weight for {layer}/{stream}: {value}"
                    )
                updated[layer][stream] = value
        self._layer_stream_weights = updated

    @torch.no_grad()
    def step(
        self,
        stream_gradients: Mapping[str, Sequence[Optional[Tensor]]],
    ) -> None:
        require_exact_streams(
            stream_gradients, self.stream_names, field="stream_gradients"
        )
        parameters = self.ordered_parameters()
        for stream in self.stream_names:
            if len(stream_gradients[stream]) != len(parameters):
                raise ValueError(
                    f"Expected {len(parameters)} gradients for {stream}, "
                    f"got {len(stream_gradients[stream])}"
                )

        parameter_index = 0
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            decay_stream = group["decay_stream"]
            for parameter in group["params"]:
                layer = self.layer_names[parameter_index]
                gradients = {
                    stream: stream_gradients[stream][parameter_index]
                    for stream in self.stream_names
                }
                parameter_index += 1
                if all(gradient is None for gradient in gradients.values()):
                    continue

                state = self.state[parameter]
                combined = None
                for stream in self.stream_names:
                    gradient = gradients[stream]
                    if gradient is not None:
                        if gradient.is_sparse:
                            raise RuntimeError(
                                "LayerwiseMultiStreamSGD does not support sparse gradients"
                            )
                        if gradient.shape != parameter.shape:
                            raise ValueError(f"Gradient shape mismatch for {layer}/{stream}")
                        gradient = gradient.detach()
                        injection_weight = self._layer_stream_weights[layer][stream]
                        # Keep the all-ones path bitwise identical to schema-v2.
                        if injection_weight != 1.0:
                            gradient = gradient.mul(injection_weight)
                    if stream == decay_stream and weight_decay != 0.0:
                        gradient = (
                            parameter.detach().mul(weight_decay)
                            if gradient is None
                            else gradient.add(parameter, alpha=weight_decay)
                        )

                    key = f"momentum_buffer_{stream}"
                    buffer = state.get(key)
                    if buffer is None:
                        if gradient is None:
                            continue
                        buffer = gradient.clone(memory_format=torch.preserve_format)
                        state[key] = buffer
                    else:
                        buffer.mul_(self._layer_momentums[layer][stream])
                        if gradient is not None:
                            buffer.add_(gradient)
                    combined = buffer.clone() if combined is None else combined.add_(buffer)

                if combined is not None and decay_stream == "outside" and weight_decay != 0.0:
                    combined.add_(parameter, alpha=weight_decay)
                if combined is not None:
                    parameter.add_(combined, alpha=-lr)

    def state_dict(self):
        state = super().state_dict()
        state["schema_version"] = CHECKPOINT_SCHEMA_VERSION
        state["stream_names"] = self.stream_names
        state["layer_names"] = self.layer_names
        state["layer_momentums"] = self.layer_momentums
        state["layer_stream_weights"] = self.layer_stream_weights
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        schema_version = state_dict.pop("schema_version", None)
        saved_streams = state_dict.pop("stream_names", None)
        if schema_version is None and saved_streams is None:
            saved_streams = LEGACY_STREAM_NAMES
        elif schema_version not in (
            LEGACY_LAYERWISE_SCHEMA_VERSION,
            CHECKPOINT_SCHEMA_VERSION,
        ):
            raise ValueError(
                f"Unsupported optimizer checkpoint schema: {schema_version!r}"
            )
        elif saved_streams is None:
            raise ValueError("Optimizer checkpoint is missing stream_names")
        saved_streams = tuple(saved_streams)
        if saved_streams != self.stream_names:
            raise ValueError(
                "Checkpoint stream order does not match optimizer: "
                f"saved {saved_streams}, current {self.stream_names}"
            )
        if "layer_names" not in state_dict or "layer_momentums" not in state_dict:
            raise ValueError("Optimizer checkpoint is missing layer-wise metadata")
        saved_layers = tuple(state_dict.pop("layer_names"))
        saved_momentums = state_dict.pop("layer_momentums")
        saved_weights = state_dict.pop("layer_stream_weights", None)
        if schema_version == CHECKPOINT_SCHEMA_VERSION and saved_weights is None:
            raise ValueError("Optimizer checkpoint is missing layer stream weights")
        if saved_layers != self.layer_names:
            raise ValueError("Checkpoint layer mapping does not match optimizer")
        result = super().load_state_dict(state_dict)
        self.set_layer_momentums(saved_momentums)
        if saved_weights is None:
            saved_weights = {
                layer: {stream: 1.0 for stream in self.stream_names}
                for layer in self._layer_stream_weights
            }
        self.set_layer_stream_weights(saved_weights)
        return result


# Compatibility import for stage-16 configs and downstream scripts. The class
# now accepts any valid ordered stream mapping; three-stream behavior is exact.
LayerwiseThreeStreamSGD = LayerwiseMultiStreamSGD
