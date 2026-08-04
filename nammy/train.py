"""
Training loop for the tinygrad WaveNet trainer.

Mirrors NAM's standard learning config: Adam(lr=0.004), ExponentialLR(gamma=0.993)
stepped per epoch, MSE loss on ny-sample windows, ESR reported on validation.
"""

from __future__ import annotations

import time

import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

from .data import Dataset
from .wavenet import WaveNet


def esr(pred: np.ndarray, target: np.ndarray) -> float:
    """Error-to-signal ratio (Wright et al. 2019, eq. 10)."""
    return float(np.sum((pred - target) ** 2) / np.sum(target**2))


def train(
    model: WaveNet,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    ny: int = 8192,
    lr: float = 0.004,
    lr_gamma: float = 0.993,
    validation_fraction: float = 0.1,
    seed: int = 0,
    out: str | None = None,
    sample_rate: float = 48000.0,
    log=print,
) -> dict:
    """
    Train the model on aligned (x, y); returns training history.

    :param out: if given, the best checkpoint so far is written there every time
        validation ESR improves, so an interrupted run still leaves a usable
        model. The extra json.dump costs ~10 ms; the weights are read back from
        the device either way to keep the in-memory best.
    """
    nx = model.receptive_field
    n_val = max(int(len(x) * validation_fraction), nx + ny)
    x_train, y_train = x[:-n_val], y[:-n_val]
    x_val, y_val = x[-n_val:], y[-n_val:]

    dataset = Dataset(x_train, y_train, nx=nx, ny=ny)
    log(
        f"receptive field {nx}, train {len(x_train)} samples "
        f"({len(dataset)} windows of {ny}), validation {len(x_val)} samples"
    )

    # The residual path of the final layer of the final array is discarded by
    # WaveNet.forward, so its 1x1 gets no gradient (same as in NAM/torch, where
    # the optimizer silently skips it). tinygrad's Adam asserts instead: exclude it.
    unused = get_parameters(model.layer_arrays[-1].layers[-1].layer1x1)
    unused_ids = {id(p) for p in unused}
    params = [p for p in get_parameters(model) if id(p) not in unused_ids]
    opt = Adam(params, lr=lr)
    rng = np.random.default_rng(seed)

    @TinyJit
    def step(xb: Tensor, yb: Tensor) -> Tensor:
        with Tensor.train():
            pred = model(xb)[:, 0, :]
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            return loss.realize()

    history = {"train_loss": [], "val_esr": [], "val_mse": []}
    best_esr, best_weights = float("inf"), None
    for epoch in range(epochs):
        t0 = time.time()
        losses = []
        for xb, yb in dataset.batches(batch_size, rng):
            loss = step(Tensor(xb), Tensor(yb))
            losses.append(float(loss.numpy()))
        opt.lr.assign(opt.lr * lr_gamma).realize()

        pred_val = model.process(x_val)
        # Skip the warmup region where the model saw zero-padding.
        v_pred, v_target = pred_val[nx - 1 :], y_val[nx - 1 :]
        val_esr = esr(v_pred, v_target)
        val_mse = float(np.mean((v_pred - v_target) ** 2))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_esr"].append(val_esr)
        history["val_mse"].append(val_mse)
        saved = ""
        if val_esr < best_esr:
            best_esr, best_weights = val_esr, model.export_weights()
            if out is not None:
                model.export_nam(out, sample_rate=sample_rate, weights=best_weights)
                saved = "  saved"
        log(
            f"epoch {epoch + 1}/{epochs}  loss {train_loss:.3e}  "
            f"val ESR {val_esr:.4f}  val MSE {val_mse:.3e}  "
            f"({time.time() - t0:.1f}s){saved}"
        )

    if best_weights is not None:
        model.import_weights(best_weights)
        log(f"restored best checkpoint (val ESR {best_esr:.4f})")
    history["best_esr"] = best_esr
    return history
