"""
POC validation:
1. receptive field / output shape
2. forward parity vs an independent numpy implementation (same weights)
3. .nam export / import_weights round trip
4. training smoke test on synthetic tanh-distortion data
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from nammy.data import Dataset
from nammy.train import esr, train
from nammy.wavenet import WaveNet, a1_config


# --- independent numpy reference (direct port of nam.models.wavenet semantics) ---


def np_conv1d(x, w, b, dilation=1):
    """Valid dilated conv. x: (Cin, L), w: (Cout, Cin, K), b: (Cout,) or None."""
    cout, cin, k = w.shape
    length = x.shape[1] - (k - 1) * dilation
    out = np.zeros((cout, length), dtype=np.float64)
    for tap in range(k):
        out += w[:, :, tap] @ x[:, tap * dilation : tap * dilation + length]
    if b is not None:
        out += b[:, None]
    return out


def np_wavenet_forward(model, x):
    """x: (L,) -> (L - R + 1,) using weights pulled from the tinygrad model."""

    def wb(conv):
        w = conv.weight.numpy().astype(np.float64)
        b = None if conv.bias is None else conv.bias.numpy().astype(np.float64)
        return w, b

    c = x[None, :]
    y, head_input = c, None
    for la in model.layer_arrays:
        out_length = min(y.shape[1], c.shape[1]) - (la.receptive_field - 1)
        w, _ = wb(la.rechannel)
        z = np_conv1d(y, w, None)
        for layer in la.layers:
            wc, bc = wb(layer.conv)
            zconv = np_conv1d(z, wc, bc, dilation=layer.dilation)
            wm, _ = wb(layer.input_mixer)
            mix = np_conv1d(c, wm, None)[:, -zconv.shape[1] :]
            post = np.tanh(zconv + mix)
            w1, b1 = wb(layer.layer1x1)
            layer_out = np_conv1d(post, w1, b1)
            head_term = post[:, -out_length:]
            head_input = (
                head_term if head_input is None else head_input[:, -out_length:] + head_term
            )
            z = z[:, -layer_out.shape[1] :] + layer_out
        wh, bh = wb(la.head_rechannel)
        head_input = np_conv1d(head_input, wh, bh)
        y = z[:, -out_length:]
    return (model.head_scale * head_input)[0]


def test_shapes():
    model = WaveNet()
    assert model.receptive_field == 4093, model.receptive_field
    from tinygrad import Tensor

    length = 5000
    out = model(Tensor(np.random.default_rng(0).normal(size=(2, 1, length)).astype(np.float32)))
    assert out.shape == (2, 1, length - 4092), out.shape
    print("PASS shapes: receptive field 4093, output (2,1,908)")


def test_numpy_parity():
    model = WaveNet()
    rng = np.random.default_rng(1)
    x = rng.normal(size=6000).astype(np.float32) * 0.5
    from tinygrad import Tensor

    got = model(Tensor(x.reshape(1, 1, -1))).numpy().flatten()
    want = np_wavenet_forward(model, x)
    err = np.max(np.abs(got - want))
    assert err < 1e-4, f"parity error {err}"
    print(f"PASS numpy parity: max abs err {err:.2e} over {len(want)} samples")


def test_export_roundtrip(tmp_path="/tmp"):
    import json

    model = WaveNet()
    weights = model.export_weights()
    # Expected count: per array: rechannel + n*(conv + mixer + 1x1) + head_rechannel; +1 head_scale
    def conv_n(cin, cout, k, bias):
        return cout * cin * k + (cout if bias else 0)

    expect = 0
    for lc in a1_config()["layers_configs"]:
        ch, k, n = lc["channels"], lc["kernel_size"], len(lc["dilations"])
        expect += conv_n(lc["input_size"], ch, 1, False)
        expect += n * (conv_n(ch, ch, k, True) + conv_n(1, ch, 1, False) + conv_n(ch, ch, 1, True))
        expect += conv_n(ch, lc["head_size"], 1, lc["head_bias"])
    expect += 1  # head_scale
    assert len(weights) == expect, (len(weights), expect)

    model2 = WaveNet()
    model2.import_weights(weights)
    assert np.allclose(model2.export_weights(), weights)

    rng = np.random.default_rng(2)
    x = rng.normal(size=5000).astype(np.float32) * 0.3
    from tinygrad import Tensor

    a = model(Tensor(x.reshape(1, 1, -1))).numpy()
    b = model2(Tensor(x.reshape(1, 1, -1))).numpy()
    # head_scale is stored as float32 in the weight blob (as in NAM), so allow
    # for its rounding: 0.02 -> 0.019999999552965164.
    assert np.allclose(a, b, atol=1e-6), "round-trip changed outputs"

    path = "/tmp/claude-1000/-home-asaka-nammy/e3c2fd2f-0653-4c11-b0ed-59a30a0ba368/scratchpad/roundtrip.nam"
    model.export_nam(path)
    with open(path) as fp:
        d = json.load(fp)
    assert d["architecture"] == "WaveNet"
    assert len(d["weights"]) == expect
    assert d["config"]["layers"][0]["channels"] == 16
    assert d["config"]["layers"][1]["head_bias"] is True
    print(f"PASS export round-trip: {expect} weights, .nam JSON OK")


def test_dataset():
    x = np.arange(100, dtype=np.float32)
    y = x + 1000
    ds = Dataset(x, y, nx=10, ny=20)
    assert len(ds) == (100 - 10 + 1) // 20
    xi, yi = ds[1]
    assert len(xi) == 10 + 20 - 1 and len(yi) == 20
    # y window must align with the last sample of each receptive field
    assert yi[0] == y[20 + 10 - 1]
    print("PASS dataset slicing/alignment")


def test_training_smoke():
    # Small config for CPU speed; same code path as A1.
    config = {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": 8,
                "head_size": 4,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32],
                "head_bias": False,
            },
            {
                "input_size": 8,
                "condition_size": 1,
                "channels": 4,
                "head_size": 1,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32],
                "head_bias": True,
            },
        ],
        "head_scale": 0.02,
    }
    model = WaveNet(config)
    rng = np.random.default_rng(3)
    x = rng.normal(size=48000).astype(np.float32) * 0.4
    y = np.tanh(3.0 * x).astype(np.float32) * 0.5  # static distortion target

    pred0 = model.process(x[-4800:])
    esr0 = esr(pred0, y[-4800:])
    history = train(
        model, x, y, epochs=5, batch_size=8, ny=1024, lr=0.004, validation_fraction=0.1, seed=0
    )
    assert history["best_esr"] < esr0, (history["best_esr"], esr0)
    assert history["best_esr"] < 0.5, history["best_esr"]
    print(f"PASS training smoke: ESR {esr0:.3f} -> {history['best_esr']:.3f}")


if __name__ == "__main__":
    test_shapes()
    test_numpy_parity()
    test_export_roundtrip()
    test_dataset()
    test_training_smoke()
    print("all POC tests passed")
