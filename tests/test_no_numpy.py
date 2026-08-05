"""
Differential test for nammy._numpy_compat.

Runs the same work twice, once against real numpy and once in a subprocess
where numpy is hidden from the import system, and compares the results. Byte
buffers have to match exactly; only the reductions are allowed to differ, since
tinygrad does not sum in the same order numpy does.
"""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, ".")

SEED = 20260805


class _HideNumpy:
    """Make `import numpy` fail, so nammy takes its no-compiled-deps path."""

    def find_spec(self, name, path=None, target=None):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("numpy is hidden for this run")
        return None


def fixture_bytes(n: int) -> bytes:
    """Deterministic bytes, generated without numpy so both sides agree."""
    import random

    return random.Random(SEED).randbytes(n)


def small_config() -> dict:
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": 8,
                "head": {"out_channels": 4, "kernel_size": 1, "bias": False},
                "kernel_sizes": 3,
                "dilations": [1, 2, 4, 8],
                "activation": "Tanh",
            },
            {
                "input_size": 8,
                "condition_size": 1,
                "channels": 4,
                "head": {"out_channels": 1, "kernel_size": 4, "bias": True},
                "kernel_sizes": 3,
                "dilations": [1, 2, 4, 8],
                "activation": "LeakyReLU",
            },
        ],
        "head_scale": 0.02,
    }


def measure(weights: list[float], label: str) -> dict:
    """
    Every value here is produced by code under test, on both sides.

    :param weights: model weights to import, so the two runs compare the same
        network rather than two random initialisations. Both sides have to
        import them: head_scale is rounded to float32 on the way through the
        weight blob, and that is worth ~1e-8 on the output on its own.
    """
    import random

    from nammy.data import Dataset, _decode_pcm, read_wav, write_wav
    from nammy.train import esr
    from nammy.wavenet import WaveNet

    out = {"which": label}

    # 1. decoding, over every width and channel count the reader accepts
    for width in (2, 3, 4):
        for channels in (1, 2):
            raw = fixture_bytes(4096 * width * channels)
            x = _decode_pcm(raw, width, channels)
            out[f"decode{width}x{channels}"] = x.tobytes().hex()

    # 2. encoding and the file round trip
    signal = _decode_pcm(fixture_bytes(3 * 8192), 3, 1)
    for width in (2, 3):
        path = os.path.join(tempfile.gettempdir(), f"nonumpy{width}.wav")
        write_wav(path, signal, 48000, sampwidth=width)
        with open(path, "rb") as fp:
            out[f"file{width}"] = fp.read().hex()
        back, rate = read_wav(path)
        out[f"reread{width}"] = [back.tobytes().hex(), rate]
        os.remove(path)

    # 3. dataset batching, which is where the shuffle and the stacking live
    y_signal = _decode_pcm(fixture_bytes(3 * 8192)[::-1], 3, 1)
    ds = Dataset(signal, y_signal, nx=64, ny=128)
    batches = list(ds.batches(4, random.Random(SEED)))
    out["batches"] = [len(batches), list(batches[0][0].shape), list(batches[0][1].shape)]
    out["batch0"] = [batches[0][0].tobytes().hex(), batches[0][1].tobytes().hex()]

    # 4. the model path: zeros, concatenate, and the one multi-axis index left
    model = WaveNet(small_config())
    model.import_weights(weights)
    processed = model.process(signal[:6000], batch_len=1024, batch_size=2)
    out["weights"] = model.export_weights()
    out["process"] = processed.tobytes().hex()

    # 5. reductions, the one place the two sides may legitimately disagree
    out["esr"] = esr(processed, y_signal[:6000])
    return out


def main():
    if "--hidden" in sys.argv:
        sys.meta_path.insert(0, _HideNumpy())
        with open(sys.argv[sys.argv.index("--weights") + 1]) as fp:
            weights = json.load(fp)
        result = measure(weights, "shim")
        import numpy  # noqa: F401  (resolves to the stand-in, via sys.modules)

        assert numpy.__name__ == "nammy._numpy_compat", numpy.__name__
        with open(sys.argv[sys.argv.index("--out") + 1], "w") as fp:
            json.dump(result, fp)
        return

    from nammy.wavenet import WaveNet

    weights = WaveNet(small_config()).export_weights()
    reference = measure(weights, "numpy")
    tmp = tempfile.mkdtemp()
    weights_path, out_path = os.path.join(tmp, "w.json"), os.path.join(tmp, "r.json")
    with open(weights_path, "w") as fp:
        json.dump(weights, fp)

    proc = subprocess.run(
        [sys.executable, __file__, "--hidden", "--weights", weights_path, "--out", out_path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise AssertionError("the run without numpy failed")
    with open(out_path) as fp:
        shim = json.load(fp)

    assert shim["which"] == "shim" and reference["which"] == "numpy"
    failures = []
    for key in reference:
        if key in ("which", "esr", "weights"):
            continue
        if reference[key] != shim[key]:
            failures.append(key)
            print(f"FAIL {key}: differs between numpy and the stand-in")

    # Reductions accumulate in a different order, so they only have to agree.
    rel = abs(reference["esr"] - shim["esr"]) / abs(reference["esr"])
    if rel > 1e-5:
        failures.append("esr")
        print(f"FAIL esr: {reference['esr']} vs {shim['esr']} (rel {rel:.2e})")

    if failures:
        raise AssertionError(f"{len(failures)} differences: {', '.join(failures)}")
    checked = [k for k in reference if k not in ("which", "weights")]
    print(f"PASS no-numpy parity: {len(checked)} results identical ({', '.join(checked)})")


def test_no_numpy_parity():
    main()


if __name__ == "__main__":
    main()
