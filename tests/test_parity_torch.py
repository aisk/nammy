"""
Correctness check against the reference PyTorch implementation
(https://github.com/sdatkinson/neural-amp-modeler): same weights -> same output.

Requires the dev dependency group (torch, pydantic) and a checkout of the
reference repo, located via the NAM_REF environment variable (defaults to a
sibling directory of this repo). Skips if either is unavailable.

The reference package's top-level __init__ pulls in pytorch_lightning etc., so
we stub the parent packages and import nam.models.wavenet._wavenet directly.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

REF = os.environ.get(
    "NAM_REF", str(Path(__file__).resolve().parents[2] / "neural-amp-modeler")
)


def _load_reference():
    if not (Path(REF) / "nam" / "models" / "wavenet" / "_wavenet.py").is_file():
        return None, f"reference repo not found at {REF} (set NAM_REF)"
    try:
        import torch  # noqa: F401
    except ImportError:
        return None, "torch not installed (uv sync --group dev)"
    for name, path in [
        ("nam", f"{REF}/nam"),
        ("nam.models", f"{REF}/nam/models"),
        ("nam.models.wavenet", f"{REF}/nam/models/wavenet"),
    ]:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod
    from nam.models.wavenet._wavenet import WaveNet as TorchWaveNet

    return TorchWaveNet, None


from nammy.wavenet import WaveNet, a1_config, a2_config  # noqa: E402


def check_arch(TorchWaveNet, name: str, config_fn, receptive_field: int) -> None:
    """nammy's configs use the reference repo's schema, so both take them as-is."""
    import torch
    from tinygrad import Tensor

    tiny = WaveNet(config_fn())
    ref = TorchWaveNet.init_from_config(config_fn())
    assert ref.receptive_field == tiny.receptive_field == receptive_field, (
        name,
        ref.receptive_field,
        tiny.receptive_field,
    )

    # tinygrad weights -> torch model
    weights = tiny.export_weights()
    used = ref.import_weights(torch.Tensor(weights))
    assert used == len(weights), (used, len(weights))

    # weight layout must match the reference export exactly
    ref_weights = ref.export_weights()
    assert len(ref_weights) == len(weights), (len(ref_weights), len(weights))
    layout_err = np.max(np.abs(ref_weights - np.asarray(weights, dtype=np.float32)))
    assert layout_err == 0.0, f"{name} weight layout mismatch, max err {layout_err}"

    # same input through both
    rng = np.random.default_rng(42)
    x = (rng.normal(size=48000) * 0.4).clip(-1, 1).astype(np.float32)

    got = tiny(Tensor(x.reshape(1, 1, -1))).numpy().flatten()
    with torch.no_grad():
        want = ref(torch.from_numpy(x).reshape(1, 1, -1)).numpy().flatten()
    assert got.shape == want.shape, (got.shape, want.shape)
    err = np.max(np.abs(got - want))
    denom = np.max(np.abs(want))
    print(f"{name} outputs: {len(got)} samples, max |ref| {denom:.3e}")
    print(f"{name} max abs diff tinygrad vs torch: {err:.3e} (rel {err / denom:.3e})")
    assert err < 1e-6, f"{name} parity failure: {err}"

    # reverse direction: torch-initialized weights -> tinygrad
    ref2 = TorchWaveNet.init_from_config(config_fn())
    tiny.import_weights(ref2.export_weights())
    got2 = tiny(Tensor(x.reshape(1, 1, -1))).numpy().flatten()
    with torch.no_grad():
        want2 = ref2(torch.from_numpy(x).reshape(1, 1, -1)).numpy().flatten()
    err2 = np.max(np.abs(got2 - want2))
    print(f"{name} reverse direction max abs diff: {err2:.3e}")
    assert err2 < 1e-6, f"{name} reverse parity failure: {err2}"

    print(f"PASS torch parity {name}")


def main():
    TorchWaveNet, skip_reason = _load_reference()
    if TorchWaveNet is None:
        print(f"SKIP torch parity: {skip_reason}")
        return

    check_arch(TorchWaveNet, "a1", a1_config, 4093)
    check_arch(TorchWaveNet, "a2", a2_config, 6347)


if __name__ == "__main__":
    main()
