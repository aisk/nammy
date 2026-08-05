"""
POC validation:
1. receptive field / output shape
2. forward parity vs an independent numpy implementation (same weights)
3. .nam export / import_weights round trip
4. training smoke test on synthetic tanh-distortion data
5. the progress/stop hooks the GUI drives training through
6. backend selection: the preference chain and explicit-device errors
"""

import os
import sys

import numpy as np

sys.path.insert(0, ".")

from tinygrad.device import Device

from nammy import device
from nammy.data import Dataset
from nammy.train import esr, train
from nammy.wavenet import WaveNet, a1_config, a2_config


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
        # The head rechannel eats (head_kernel - 1) samples after the layer sum.
        pre_head = min(y.shape[1], c.shape[1]) - (la.receptive_field - la.head_kernel)
        act = {
            "Tanh": np.tanh,
            "LeakyReLU": lambda v: np.where(v > 0, v, 0.01 * v),
        }[la._config["activation"]]
        w, _ = wb(la.rechannel)
        z = np_conv1d(y, w, None)
        for layer in la.layers:
            wc, bc = wb(layer.conv)
            zconv = np_conv1d(z, wc, bc, dilation=layer.dilation)
            wm, _ = wb(layer.input_mixer)
            mix = np_conv1d(c, wm, None)[:, -zconv.shape[1] :]
            post = act(zconv + mix)
            w1, b1 = wb(layer.layer1x1)
            layer_out = np_conv1d(post, w1, b1)
            head_term = post[:, -pre_head:]
            head_input = (
                head_term if head_input is None else head_input[:, -pre_head:] + head_term
            )
            z = z[:, -layer_out.shape[1] :] + layer_out
        wh, bh = wb(la.head_rechannel)
        head_input = np_conv1d(head_input, wh, bh)
        y = z[:, -(pre_head - (la.head_kernel - 1)) :]
    return (model.head_scale * head_input)[0]


def test_shapes():
    from tinygrad import Tensor

    for name, config, rf in [("a1", a1_config(), 4093), ("a2", a2_config(), 6347)]:
        model = WaveNet(config)
        assert model.receptive_field == rf, (name, model.receptive_field)
        length = rf + 907
        out = model(
            Tensor(np.random.default_rng(0).normal(size=(2, 1, length)).astype(np.float32))
        )
        assert out.shape == (2, 1, 908), (name, out.shape)
        print(f"PASS shapes {name}: receptive field {rf}, output (2,1,908)")
    assert WaveNet().receptive_field == 6347  # the default is a2


def test_numpy_parity():
    from tinygrad import Tensor

    for name, config in [("a1", a1_config()), ("a2", a2_config())]:
        model = WaveNet(config)
        rng = np.random.default_rng(1)
        x = rng.normal(size=8000).astype(np.float32) * 0.5
        got = model(Tensor(x.reshape(1, 1, -1))).numpy().flatten()
        want = np_wavenet_forward(model, x)
        err = np.max(np.abs(got - want))
        assert err < 1e-4, f"{name} parity error {err}"
        print(f"PASS numpy parity {name}: max abs err {err:.2e} over {len(want)} samples")


def _expected_weight_count(config: dict) -> int:
    """rechannel + per layer (conv + mixer + 1x1) + head_rechannel, + head_scale."""

    def conv_n(cin, cout, k, bias):
        return cout * cin * k + (cout if bias else 0)

    expect = 0
    for lc in config["layers_configs"]:
        ch, head = lc["channels"], lc["head"]
        ks = lc["kernel_sizes"]
        ks = [ks] * len(lc["dilations"]) if isinstance(ks, int) else ks
        expect += conv_n(lc["input_size"], ch, 1, False)
        expect += sum(
            conv_n(ch, ch, k, True) + conv_n(1, ch, 1, False) + conv_n(ch, ch, 1, True)
            for k in ks
        )
        expect += conv_n(ch, head["out_channels"], head["kernel_size"], head["bias"])
    return expect + 1  # head_scale


def test_export_roundtrip(tmp_path=None):
    import json
    import tempfile

    from tinygrad import Tensor

    for name, config in [("a1", a1_config()), ("a2", a2_config())]:
        model = WaveNet(config)
        weights = model.export_weights()
        expect = _expected_weight_count(config)
        assert len(weights) == expect, (name, len(weights), expect)

        model2 = WaveNet(config)
        model2.import_weights(weights)
        assert np.allclose(model2.export_weights(), weights)

        rng = np.random.default_rng(2)
        x = rng.normal(size=8000).astype(np.float32) * 0.3
        a = model(Tensor(x.reshape(1, 1, -1))).numpy()
        b = model2(Tensor(x.reshape(1, 1, -1))).numpy()
        # head_scale is stored as float32 in the weight blob (as in NAM), so allow
        # for its rounding: 0.02 -> 0.019999999552965164.
        assert np.allclose(a, b, atol=1e-6), f"{name} round-trip changed outputs"

        path = os.path.join(tmp_path or tempfile.gettempdir(), f"roundtrip-{name}.nam")
        model.export_nam(path)
        with open(path) as fp:
            d = json.load(fp)
        assert d["architecture"] == "WaveNet"
        assert len(d["weights"]) == expect
        layer0 = d["config"]["layers"][0]
        if name == "a1":
            # A1 fits the classic schema, which every plugin version can read.
            assert d["version"] == "0.5.4", d["version"]
            assert layer0["channels"] == 16 and layer0["kernel_size"] == 3
            assert d["config"]["layers"][1]["head_bias"] is True
        else:
            assert d["version"] == "0.7.0", d["version"]
            assert layer0["kernel_sizes"] == [6] * 14 + [15, 15] + [6] * 7
            assert layer0["head"] == {"out_channels": 1, "kernel_size": 16, "bias": True}
            assert layer0["activation"][0] == {"type": "LeakyReLU", "negative_slope": 0.01}
            assert layer0["gating_mode"] == ["none"] * 23

        # Loading the file back must reproduce the model (covers both schemas).
        loaded, rate = WaveNet.from_nam(path)
        assert rate == 48000.0
        assert loaded.receptive_field == model.receptive_field
        assert np.allclose(loaded.export_weights(), weights)
        print(f"PASS export round-trip {name}: {expect} weights, .nam JSON OK")


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


def test_device_selection():
    chain = device.preference()
    assert chain[0] == ("METAL" if sys.platform == "darwin" else "CL"), chain
    assert chain[-1].startswith("CPU"), chain
    assert len(set(chain)) == len(chain), chain

    # An explicit device that cannot work is an error, never a quiet fallback.
    assert device.probe("NOT_A_REAL_DEVICE") is not None
    try:
        device.select("NOT_A_REAL_DEVICE")
        raise AssertionError("an impossible device should not be selectable")
    except device.DeviceError as exc:
        assert "not usable" in str(exc), exc

    target = device.select()
    assert device.current() == target
    assert Device.DEFAULT == target.split(":")[0], (Device.DEFAULT, target)
    if device._USER_SPLIT_THRESHOLD is None:
        # The reduce split is a GPU win and a CPU loss, so it tracks the device.
        threshold = os.environ.get("REDUCEOP_SPLIT_THRESHOLD")
        assert (threshold is None) == target.startswith("CPU"), (target, threshold)
    print(f"PASS device selection: {' -> '.join(chain)} picked {target}")


def small_config():
    """Small config for CPU speed; A1's shape with A2's head conv and activation."""
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": 8,
                "head": {"out_channels": 4, "kernel_size": 1, "bias": False},
                "kernel_sizes": 3,
                "dilations": [1, 2, 4, 8, 16, 32],
                "activation": "Tanh",
            },
            {
                "input_size": 8,
                "condition_size": 1,
                "channels": 4,
                "head": {"out_channels": 1, "kernel_size": 4, "bias": True},
                "kernel_sizes": 3,
                "dilations": [1, 2, 4, 8, 16, 32],
                "activation": "LeakyReLU",
            },
        ],
        "head_scale": 0.02,
    }


def test_training_smoke():
    model = WaveNet(small_config())
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


def test_train_hooks():
    """The GUI drives training through should_stop/on_epoch/on_batch."""
    model = WaveNet(small_config())
    rng = np.random.default_rng(3)
    x = rng.normal(size=48000).astype(np.float32) * 0.4
    y = np.tanh(3.0 * x).astype(np.float32) * 0.5

    epochs, batch_size, ny = 4, 8, 1024
    seen_epochs, seen_batches = [], []
    history = train(
        model,
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        ny=ny,
        validation_fraction=0.1,
        on_epoch=seen_epochs.append,
        on_batch=lambda done, total: seen_batches.append((done, total)),
        log=lambda _m: None,
    )
    assert not history["stopped"]
    assert len(seen_epochs) == epochs, len(seen_epochs)
    assert [e["epoch"] for e in seen_epochs] == list(range(1, epochs + 1))
    assert seen_epochs[-1]["val_esr"] == history["val_esr"][-1]
    assert seen_epochs[-1]["best_esr"] == history["best_esr"]
    # on_batch counts up to the batch count the dataset actually yields.
    n_val = max(int(len(x) * 0.1), model.receptive_field + ny)
    expected = Dataset(x[:-n_val], y[:-n_val], nx=model.receptive_field, ny=ny).n_batches(
        batch_size
    )
    assert seen_batches[expected - 1] == (expected, expected), seen_batches[:3]
    assert len(seen_batches) == expected * epochs, (len(seen_batches), expected)

    # process() reports progress the same way, for the GUI's other progress bar.
    steps = []
    model.process(
        x[:5000], batch_len=1024, batch_size=2, progress=lambda d, t: steps.append((d, t))
    )
    assert steps[-1] == (5000, 5000), steps

    # Stopping mid-epoch ends the run without recording that epoch.
    calls = []
    stopped = train(
        WaveNet(small_config()),
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        ny=ny,
        validation_fraction=0.1,
        should_stop=lambda: len(calls) >= 2,
        on_batch=lambda done, _t: calls.append(done),
        log=lambda _m: None,
    )
    assert stopped["stopped"] is True
    assert len(calls) == 2, calls
    assert stopped["val_esr"] == [], stopped["val_esr"]
    print(f"PASS train hooks: {expected} batches/epoch, stop honored after {len(calls)}")


if __name__ == "__main__":
    test_device_selection()
    test_shapes()
    test_numpy_parity()
    test_export_roundtrip()
    test_dataset()
    test_training_smoke()
    test_train_hooks()
    print("all POC tests passed")
