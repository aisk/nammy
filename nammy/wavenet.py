"""
NAM WaveNet (the "A1" and "A2" architectures) implemented with tinygrad.

Faithful port of neural-amp-modeler's nam.models.wavenet, covering the feature
set the standard architectures use: dilated conv + condition input mixer +
activation + residual 1x1, per-array head rechannel (a plain causal conv, 1x1
in A1 and 16-tap in A2), no gating/FiLM/head-1x1, top-level head disabled.
"""

from __future__ import annotations

import json
import math
import os
import struct
from datetime import datetime
from typing import Callable, Optional, Sequence

try:
    import numpy as np
except ImportError:  # a build with no compiled dependencies
    from . import _numpy_compat as np
from tinygrad import Tensor, TinyJit, dtypes, nn

from .device import ensure_selected

# A1-shaped models (uniform kernel, 1x1 head rechannel) export the classic
# schema every NeuralAmpModelerCore can read; A2 needs the current one.
_EXPORT_VERSION_CLASSIC = "0.5.4"
_EXPORT_VERSION = "0.7.0"

# torch's LeakyReLU default negative_slope, which is what NAM trains with.
_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "Tanh": Tensor.tanh,
    "LeakyReLU": lambda t: t.leaky_relu(0.01),
}

_FILM_KEYS = (
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
)


def a1_config() -> dict:
    """The classic standard WaveNet (nam_full_configs/models/wavenet.json)."""
    dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    def layer(input_size: int, channels: int, head_out: int, head_bias: bool) -> dict:
        return {
            "input_size": input_size,
            "condition_size": 1,
            "channels": channels,
            "head": {"out_channels": head_out, "kernel_size": 1, "bias": head_bias},
            "kernel_sizes": 3,
            "dilations": list(dilations),
            "activation": "Tanh",
        }

    return {
        "layers_configs": [layer(1, 16, 8, False), layer(16, 8, 1, True)],
        "head_scale": 0.02,
    }


def a2_config(channels: int = 8) -> dict:
    """
    The current default architecture (nam/train/_resources/config_model_packed.json):
    one LeakyReLU layer array with restarting dilations, two 15-tap layers among
    the 6-tap ones, and a 16-tap head rechannel. The reference trainer packs the
    8-channel ("standard") and 3-channel ("lite") widths into one training run;
    here they are simply two values of `channels`.
    """
    dilations = [1, 3, 7, 17, 41, 101, 239]
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": channels,
                "head": {"out_channels": 1, "kernel_size": 16, "bias": True},
                "kernel_sizes": [6] * 14 + [15, 15] + [6] * 7,
                "dilations": dilations * 2 + [1, 13] + dilations,
                "activation": "LeakyReLU",
            }
        ],
        "head_scale": 0.01,
    }


# The selectable presets, in the order UIs should offer them.
ARCHITECTURES: dict[str, Callable[[], dict]] = {
    "a2": a2_config,
    "a2-lite": lambda: a2_config(channels=3),
    "a1": a1_config,
}


class _Layer:
    def __init__(
        self,
        condition_size: int,
        channels: int,
        kernel_size: int,
        dilation: int,
        activation: str,
    ):
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=True)
        self.input_mixer = nn.Conv1d(condition_size, channels, 1, bias=False)
        self.layer1x1 = nn.Conv1d(channels, channels, 1, bias=True)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = _ACTIVATIONS[activation]

    def __call__(self, x: Tensor, h: Tensor, out_length: int) -> tuple[Tensor, Tensor]:
        """
        :param x: (B,C,L1) from previous layer
        :param h: (B,DX,L2) condition
        :return: (residual to next layer, head term of length out_length)
        """
        zconv = self.conv(x)
        mix = self.input_mixer(h)[:, :, -zconv.shape[2] :]
        post_activation = self.activation(zconv + mix)
        layer_output = self.layer1x1(post_activation)
        head_output = post_activation[:, :, -out_length:]
        residual = x[:, :, -layer_output.shape[2] :] + layer_output
        return residual, head_output


class _LayerArray:
    def __init__(
        self,
        input_size: int,
        condition_size: int,
        channels: int,
        head: dict,
        kernel_sizes: int | Sequence[int],
        dilations: list[int],
        activation: str,
    ):
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(dilations)
        assert len(kernel_sizes) == len(dilations), (kernel_sizes, dilations)
        self.rechannel = nn.Conv1d(input_size, channels, 1, bias=False)
        self.layers = [
            _Layer(condition_size, channels, k, d, activation)
            for k, d in zip(kernel_sizes, dilations)
        ]
        self.head_kernel = int(head["kernel_size"])
        self.head_rechannel = nn.Conv1d(
            channels, head["out_channels"], head["kernel_size"], bias=head["bias"]
        )
        self._config = {
            "input_size": input_size,
            "condition_size": condition_size,
            "channels": channels,
            "head": dict(head),
            "kernel_sizes": list(kernel_sizes),
            "dilations": list(dilations),
            "activation": activation,
        }

    @property
    def receptive_field(self) -> int:
        layers = 1 + sum((layer.kernel_size - 1) * layer.dilation for layer in self.layers)
        return layers + self.head_kernel - 1

    def __call__(
        self, x: Tensor, c: Tensor, head_input: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        # The head rechannel is a valid conv too, so the head sum runs at a
        # length that leaves it (head_kernel - 1) samples to consume.
        pre_head = min(x.shape[2], c.shape[2]) - (self.receptive_field - self.head_kernel)
        x = self.rechannel(x)
        for layer in self.layers:
            x, head_term = layer(x, c, pre_head)
            head_input = (
                head_term if head_input is None else head_input[:, :, -pre_head:] + head_term
            )
        out_length = pre_head - (self.head_kernel - 1)
        return self.head_rechannel(head_input), x[:, :, -out_length:]

    @property
    def classic_exportable(self) -> bool:
        cfg = self._config
        return len(set(cfg["kernel_sizes"])) == 1 and cfg["head"]["kernel_size"] == 1

    def export_config_classic(self) -> dict:
        """Per-array config in the classic (v0.5.x) schema."""
        cfg = self._config
        assert self.classic_exportable
        return {
            "input_size": cfg["input_size"],
            "condition_size": cfg["condition_size"],
            "channels": cfg["channels"],
            "head_size": cfg["head"]["out_channels"],
            "kernel_size": cfg["kernel_sizes"][0],
            "dilations": list(cfg["dilations"]),
            "head_bias": cfg["head"]["bias"],
            "activation": cfg["activation"],
            "gated": False,
        }

    def export_config(self) -> dict:
        """Per-array config in the current schema, optional features all off."""
        cfg = self._config
        n = len(cfg["dilations"])
        activation: dict = {"type": cfg["activation"]}
        if cfg["activation"] == "LeakyReLU":
            activation["negative_slope"] = 0.01
        film_off = {"active": False, "shift": True, "groups": 1}
        return {
            "input_size": cfg["input_size"],
            "condition_size": cfg["condition_size"],
            "head": dict(cfg["head"]),
            "channels": cfg["channels"],
            "kernel_sizes": list(cfg["kernel_sizes"]),
            "dilations": list(cfg["dilations"]),
            "activation": [dict(activation) for _ in range(n)],
            "bottleneck": cfg["channels"],
            "head1x1": {"active": False, "out_channels": 1, "groups": 1},
            "layer1x1": {"active": True, "groups": 1},
            "groups_input": 1,
            "groups_input_mixin": 1,
            **{key: dict(film_off) for key in _FILM_KEYS},
            "gating_mode": ["none"] * n,
            "secondary_activation": [None] * n,
            "slimmable": None,
        }

    def _weight_modules(self):
        mods = [self.rechannel]
        for layer in self.layers:
            mods.extend([layer.conv, layer.input_mixer, layer.layer1x1])
        mods.append(self.head_rechannel)
        return mods

    def export_weights(self) -> list[float]:
        return [v for m in self._weight_modules() for v in _conv_weights(m)]

    def import_weights(self, weights: Sequence[float], i: int) -> int:
        for m in self._weight_modules():
            i = _conv_import(m, weights, i)
        return i


def _conv_weights(conv) -> list[float]:
    values = conv.weight.flatten().tolist()
    if conv.bias is not None:
        values += conv.bias.flatten().tolist()
    return values


def _conv_import(conv, weights: Sequence[float], i: int) -> int:
    for param in (conv.weight, conv.bias):
        if param is None:
            continue
        n = math.prod(param.shape)
        param.assign(
            Tensor(weights[i : i + n], dtype=dtypes.float32).reshape(param.shape)
        ).realize()
        i += n
    return i


def _activation_name(a) -> str:
    """The activation's name from either the .nam form ({"type": ...}) or a string."""
    if isinstance(a, dict):
        if a.get("type") == "LeakyReLU" and a.get("negative_slope", 0.01) != 0.01:
            raise NotImplementedError(
                f"LeakyReLU negative_slope {a['negative_slope']} is not supported"
            )
        return a.get("type", a.get("name"))
    return a


def _layer_config_from_nam(lc: dict) -> dict:
    """Normalize one .nam layer-array config (classic or current schema)."""
    if lc.get("gated") or any(mode != "none" for mode in lc.get("gating_mode", [])):
        raise NotImplementedError("gated WaveNet layers are not supported")

    activation = lc.get("activation", "Tanh")
    names = {_activation_name(a) for a in activation} if isinstance(activation, list) else {
        _activation_name(activation)
    }
    if len(names) != 1:
        raise NotImplementedError(f"per-layer activations are not supported: {sorted(names)}")
    activation = names.pop()
    if activation not in _ACTIVATIONS:
        raise NotImplementedError(f"activation {activation!r} is not supported")

    if "head" in lc:  # current schema (v0.6+)
        unsupported = {
            "head1x1": lc.get("head1x1", {}).get("active", False),
            "disabled layer1x1": not lc.get("layer1x1", {}).get("active", True),
            "FiLM": any(lc.get(key, {}).get("active", False) for key in _FILM_KEYS),
            "bottleneck": lc.get("bottleneck", lc["channels"]) != lc["channels"],
            "grouped convs": lc.get("groups_input", 1) != 1
            or lc.get("groups_input_mixin", 1) != 1
            or lc.get("layer1x1", {}).get("groups", 1) != 1,
            "slimmable": lc.get("slimmable") is not None,
        }
        for feature, active in unsupported.items():
            if active:
                raise NotImplementedError(f"WaveNet with {feature} is not supported")
        head = lc["head"]
        kernel_sizes = lc["kernel_sizes"] if "kernel_sizes" in lc else lc["kernel_size"]
    else:  # classic schema (v0.5.x)
        head = {"out_channels": lc["head_size"], "kernel_size": 1, "bias": lc["head_bias"]}
        kernel_sizes = lc["kernel_size"]

    return {
        "input_size": lc["input_size"],
        "condition_size": lc["condition_size"],
        "channels": lc["channels"],
        "head": {
            "out_channels": head["out_channels"],
            "kernel_size": head["kernel_size"],
            "bias": head["bias"],
        },
        "kernel_sizes": kernel_sizes,
        "dilations": list(lc["dilations"]),
        "activation": activation,
    }


class WaveNet:
    @classmethod
    def from_nam(cls, path: str) -> tuple["WaveNet", Optional[float]]:
        """Load a WaveNet .nam file (either schema); returns (model, sample_rate)."""
        with open(path) as fp:
            d = json.load(fp)
        if d.get("architecture") != "WaveNet":
            raise ValueError(f"Unsupported architecture: {d.get('architecture')!r}")
        cfg = d["config"]
        if cfg.get("head") is not None:
            raise NotImplementedError("WaveNet with a head module is not supported")
        if cfg.get("condition_dsp") is not None:
            raise NotImplementedError("WaveNet with a condition DSP is not supported")
        layers_configs = [_layer_config_from_nam(lc) for lc in cfg["layers"]]
        model = cls({"layers_configs": layers_configs, "head_scale": cfg["head_scale"]})
        model.import_weights(d["weights"])
        return model, d.get("sample_rate")

    def __init__(self, config: Optional[dict] = None):
        # Building the model allocates tensors, so the backend has to be settled
        # by now; this is a no-op once anything has called device.select().
        ensure_selected()
        config = config if config is not None else a2_config()
        self.layer_arrays = [_LayerArray(**lc) for lc in config["layers_configs"]]
        self.head_scale = float(config["head_scale"])
        self._jits: dict[tuple[int, int], TinyJit] = {}

    @property
    def receptive_field(self) -> int:
        return 1 + sum(la.receptive_field - 1 for la in self.layer_arrays)

    def __call__(self, x: Tensor) -> Tensor:
        """
        :param x: (B,1,L)
        :return: (B,1,L-(R-1))
        """
        c = x
        y, head_input = x, None
        for layer_array in self.layer_arrays:
            head_input, y = layer_array(y, c, head_input)
        return head_input * self.head_scale

    def _jit_forward(self, batch_size: int, length: int) -> TinyJit:
        """A jitted forward for one input shape, kept so it survives across calls."""
        key = (batch_size, length)
        if key not in self._jits:
            self._jits[key] = TinyJit(lambda t: self(t).realize())
        return self._jits[key]

    def process(
        self,
        x: np.ndarray,
        batch_len: int = 65536,
        batch_size: int = 16,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Run a 1D signal through the model, padding so output aligns with input.

        Chunks each carry their own receptive-field prefix and are therefore
        independent, so they go through the network stacked on the batch axis.
        One chunk at a time leaves the kernels far too small to fill the GPU and
        rebuilds the graph in Python for every chunk: batching under a JIT is 11x
        faster on the same signal (5.33s -> 0.47s for a 1.5M-sample validation set).

        :param progress: called with (samples done, total) after each batch.
        """
        pad = self.receptive_field - 1
        x_padded = np.concatenate([np.zeros(pad, dtype=np.float32), x.astype(np.float32)])
        starts = range(0, len(x), batch_len)
        fwd = self._jit_forward(batch_size, batch_len + pad)
        # A partly filled last group still runs at the jitted shape; the network is
        # causal, so the zero tail cannot reach the outputs we keep.
        buf = np.zeros((batch_size, 1, batch_len + pad), dtype=np.float32)
        out = np.zeros(len(x), dtype=np.float32)
        for i in range(0, len(starts), batch_size):
            group = starts[i : i + batch_size]
            buf[:] = 0.0
            for row, start in enumerate(group):
                chunk = x_padded[start : start + batch_len + pad]
                buf[row, 0, : len(chunk)] = chunk
            y = fwd(Tensor(buf)).numpy()
            for row, start in enumerate(group):
                n = min(batch_len, len(x) - start)
                out[start : start + n] = y[row, 0, :n]
            if progress is not None:
                progress(min(group[-1] + batch_len, len(x)), len(x))
        return out

    def export_weights(self) -> list[float]:
        weights = [v for la in self.layer_arrays for v in la.export_weights()]
        # Every other weight comes back out of a float32 tensor, and NAM keeps
        # head_scale in the same float32 blob, so it is rounded the same way.
        # The torch parity test compares the two exports for exact equality.
        weights.append(struct.unpack("<f", struct.pack("<f", self.head_scale))[0])
        return weights

    def import_weights(self, weights: Sequence[float]) -> None:
        # head_scale is baked into a jitted graph as a constant, so drop the jits.
        # (The conv weights are assigned in place and stay valid, but this is rare
        # enough that re-tracing is cheaper than reasoning about it.)
        self._jits.clear()
        i = 0
        for la in self.layer_arrays:
            i = la.import_weights(weights, i)
        self.head_scale = float(weights[i])
        i += 1
        assert i == len(weights), f"weight count mismatch: used {i} of {len(weights)}"

    @property
    def _classic_export(self) -> bool:
        return all(la.classic_exportable for la in self.layer_arrays)

    def export_config(self) -> dict:
        classic = self._classic_export
        return {
            "layers": [
                la.export_config_classic() if classic else la.export_config()
                for la in self.layer_arrays
            ],
            "head": None,
            "head_scale": self.head_scale,
        }

    def export_nam(
        self,
        path: str,
        sample_rate: float = 48000.0,
        metadata: Optional[dict] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Write a .nam file: the classic schema when the model fits it (so A1
        models stay loadable everywhere), the current schema otherwise.

        :param weights: weights to write instead of the model's current ones,
            so a checkpoint can be saved without importing it back first.
        """
        t = datetime.now()
        model_dict = {
            "version": _EXPORT_VERSION_CLASSIC if self._classic_export else _EXPORT_VERSION,
            "metadata": {
                "date": {
                    "year": t.year,
                    "month": t.month,
                    "day": t.day,
                    "hour": t.hour,
                    "minute": t.minute,
                    "second": t.second,
                },
                "loudness": None,
                "gain": None,
                **(metadata or {}),
            },
            "architecture": "WaveNet",
            "config": self.export_config(),
            "weights": self.export_weights() if weights is None else list(weights),
            "sample_rate": sample_rate,
        }
        # Write-then-rename: checkpoints land mid-training, and an interrupt
        # during the write must not leave a truncated .nam behind.
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fp:
            json.dump(model_dict, fp)
        os.replace(tmp, path)
