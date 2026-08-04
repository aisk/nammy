"""
NAM WaveNet (classic "A1" architecture) implemented with tinygrad.

Faithful port of neural-amp-modeler's nam.models.wavenet with the A1 feature
set: dilated conv + condition input mixer + Tanh + residual 1x1, per-array
head rechannel, no gating/FiLM/head-1x1, top-level head disabled.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
from tinygrad import Tensor, TinyJit, nn

# Exported .nam files use the classic schema understood by NeuralAmpModelerCore.
_EXPORT_VERSION = "0.5.4"


def a1_config() -> dict:
    """The standard WaveNet config (nam_full_configs/models/wavenet.json)."""
    dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": 16,
                "head_size": 8,
                "kernel_size": 3,
                "dilations": list(dilations),
                "head_bias": False,
            },
            {
                "input_size": 16,
                "condition_size": 1,
                "channels": 8,
                "head_size": 1,
                "kernel_size": 3,
                "dilations": list(dilations),
                "head_bias": True,
            },
        ],
        "head_scale": 0.02,
    }


class _Layer:
    def __init__(self, condition_size: int, channels: int, kernel_size: int, dilation: int):
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=True)
        self.input_mixer = nn.Conv1d(condition_size, channels, 1, bias=False)
        self.layer1x1 = nn.Conv1d(channels, channels, 1, bias=True)
        self.kernel_size = kernel_size
        self.dilation = dilation

    def __call__(self, x: Tensor, h: Tensor, out_length: int) -> tuple[Tensor, Tensor]:
        """
        :param x: (B,C,L1) from previous layer
        :param h: (B,DX,L2) condition
        :return: (residual to next layer, head term of length out_length)
        """
        zconv = self.conv(x)
        mix = self.input_mixer(h)[:, :, -zconv.shape[2] :]
        post_activation = (zconv + mix).tanh()
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
        head_size: int,
        kernel_size: int,
        dilations: list[int],
        head_bias: bool,
    ):
        self.rechannel = nn.Conv1d(input_size, channels, 1, bias=False)
        self.layers = [_Layer(condition_size, channels, kernel_size, d) for d in dilations]
        self.head_rechannel = nn.Conv1d(channels, head_size, 1, bias=head_bias)
        self._config = {
            "input_size": input_size,
            "condition_size": condition_size,
            "channels": channels,
            "head_size": head_size,
            "kernel_size": kernel_size,
            "dilations": list(dilations),
            "head_bias": head_bias,
        }

    @property
    def receptive_field(self) -> int:
        return 1 + sum((layer.kernel_size - 1) * layer.dilation for layer in self.layers)

    def __call__(
        self, x: Tensor, c: Tensor, head_input: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        out_length = min(x.shape[2], c.shape[2]) - (self.receptive_field - 1)
        x = self.rechannel(x)
        for layer in self.layers:
            x, head_term = layer(x, c, out_length)
            head_input = (
                head_term if head_input is None else head_input[:, :, -out_length:] + head_term
            )
        return self.head_rechannel(head_input), x[:, :, -out_length:]

    def export_config(self) -> dict:
        cfg = dict(self._config)
        cfg["activation"] = "Tanh"
        cfg["gated"] = False
        return cfg

    def _weight_modules(self):
        mods = [self.rechannel]
        for layer in self.layers:
            mods.extend([layer.conv, layer.input_mixer, layer.layer1x1])
        mods.append(self.head_rechannel)
        return mods

    def export_weights(self) -> np.ndarray:
        return np.concatenate([_conv_weights(m) for m in self._weight_modules()])

    def import_weights(self, weights: np.ndarray, i: int) -> int:
        for m in self._weight_modules():
            i = _conv_import(m, weights, i)
        return i


def _conv_weights(conv) -> np.ndarray:
    tensors = [conv.weight.numpy().flatten()]
    if conv.bias is not None:
        tensors.append(conv.bias.numpy().flatten())
    return np.concatenate(tensors)


def _conv_import(conv, weights: np.ndarray, i: int) -> int:
    n = int(np.prod(conv.weight.shape))
    conv.weight.assign(
        Tensor(weights[i : i + n].astype(np.float32).reshape(tuple(conv.weight.shape)))
    ).realize()
    i += n
    if conv.bias is not None:
        n = int(np.prod(conv.bias.shape))
        conv.bias.assign(
            Tensor(weights[i : i + n].astype(np.float32).reshape(tuple(conv.bias.shape)))
        ).realize()
        i += n
    return i


class WaveNet:
    @classmethod
    def from_nam(cls, path: str) -> tuple["WaveNet", Optional[float]]:
        """Load a classic-schema WaveNet .nam file; returns (model, sample_rate)."""
        with open(path) as fp:
            d = json.load(fp)
        if d.get("architecture") != "WaveNet":
            raise ValueError(f"Unsupported architecture: {d.get('architecture')!r}")
        cfg = d["config"]
        if cfg.get("head") is not None:
            raise NotImplementedError("WaveNet with a head module is not supported")
        layers_configs = []
        for lc in cfg["layers"]:
            if lc.get("activation") != "Tanh" or lc.get("gated"):
                raise NotImplementedError(
                    f"Only Tanh/non-gated layers supported, got {lc.get('activation')!r}"
                    f"{' gated' if lc.get('gated') else ''}"
                )
            layers_configs.append(
                {
                    "input_size": lc["input_size"],
                    "condition_size": lc["condition_size"],
                    "channels": lc["channels"],
                    "head_size": lc["head_size"],
                    "kernel_size": lc["kernel_size"],
                    "dilations": list(lc["dilations"]),
                    "head_bias": lc["head_bias"],
                }
            )
        model = cls({"layers_configs": layers_configs, "head_scale": cfg["head_scale"]})
        model.import_weights(np.asarray(d["weights"], dtype=np.float64))
        return model, d.get("sample_rate")

    def __init__(self, config: Optional[dict] = None):
        config = config if config is not None else a1_config()
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

    def process(self, x: np.ndarray, batch_len: int = 65536, batch_size: int = 16) -> np.ndarray:
        """
        Run a 1D signal through the model, padding so output aligns with input.

        Chunks each carry their own receptive-field prefix and are therefore
        independent, so they go through the network stacked on the batch axis.
        One chunk at a time leaves the kernels far too small to fill the GPU and
        rebuilds the graph in Python for every chunk: batching under a JIT is 11x
        faster on the same signal (5.33s -> 0.47s for a 1.5M-sample validation set).
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
        return out

    def export_weights(self) -> np.ndarray:
        weights = np.concatenate([la.export_weights() for la in self.layer_arrays])
        return np.concatenate([weights, np.array([self.head_scale], dtype=np.float32)])

    def import_weights(self, weights: np.ndarray) -> None:
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

    def export_config(self) -> dict:
        return {
            "layers": [la.export_config() for la in self.layer_arrays],
            "head": None,
            "head_scale": self.head_scale,
        }

    def export_nam(
        self,
        path: str,
        sample_rate: float = 48000.0,
        metadata: Optional[dict] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Write a classic-schema .nam file.

        :param weights: weights to write instead of the model's current ones,
            so a checkpoint can be saved without importing it back first.
        """
        t = datetime.now()
        model_dict = {
            "version": _EXPORT_VERSION,
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
            "weights": (self.export_weights() if weights is None else weights)
            .astype(float)
            .tolist(),
            "sample_rate": sample_rate,
        }
        # Write-then-rename: checkpoints land mid-training, and an interrupt
        # during the write must not leave a truncated .nam behind.
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fp:
            json.dump(model_dict, fp)
        os.replace(tmp, path)
