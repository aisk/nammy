# nammy

A proof-of-concept trainer for [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)'s
classic **A1 WaveNet architecture**, implemented with [tinygrad](https://github.com/tinygrad/tinygrad)
instead of PyTorch.

## What's implemented

- The standard A1 WaveNet: two layer arrays (16 and 8 channels), dilations
  1–512, kernel size 3, Tanh, residual 1x1s, per-array head rechannel,
  `head_scale = 0.02`, receptive field 4093 — a faithful port of
  `nam.models.wavenet`.
- NAM-style data pipeline: WAV loading (PCM 16/24/32 and IEEE float),
  latency compensation, and `(nx+ny-1, ny)` window slicing matching
  `nam.data.Dataset`.
- Training matching NAM's standard learning config: Adam(lr=0.004),
  per-epoch exponential LR decay (gamma 0.993), MSE loss, ESR validation with
  best-checkpoint restore, JIT-compiled train step.
- Export to `.nam` (classic v0.5.4 schema) loadable by the NAM plugin.

Not implemented (yet): A2/packed training, gated/FiLM variants, MRSTFT loss,
pre-emphasis, output loudness normalization, the standardized input-file
splits/checks.

## Usage

Train from an input/output pair:

```console
$ uv run python -m nammy train input.wav output.wav --epochs 100 --out model.nam
```

`input.wav` is the DI/reamp source, `output.wav` the processed capture; both
must share a sample rate and be time-aligned (use `--latency <samples>` to
compensate reamp latency).

Run audio through a trained model (reamp):

```console
$ uv run python -m nammy process model.nam input.wav output.wav
```

This also loads classic-schema (non-gated Tanh WaveNet) `.nam` files trained
elsewhere.

## Example training run

Reference numbers from a run on consumer hardware, training on the Blackstar
HT-1 capture pair from [Alec Wright's dataset](https://github.com/Alec-Wright/Automated-GuitarAmpModelling)
(340 s of aligned input/target at 44.1 kHz):

- **Hardware**: AMD Radeon RX 6800 on Windows 11, via tinygrad's OpenCL (`CL`)
  backend.
- **Config**: defaults — 100 epochs, batch 16, `ny` 8192, Adam(lr=0.004).
- **Speed**: 88 s for the first epoch (dominated by JIT compilation), then a
  steady ~26 s per epoch; ~44 min total.
- **Result**: best validation ESR **0.0065**, reached at epoch 96. For
  reference, ESR below 0.01 is a good model and 0.02–0.05 is usable.

Convergence is fast: ESR hits 0.042 by epoch 3 and 0.0094 by epoch 40, then
flattens, with the last 20 epochs improving it by under 2% while the training
loss keeps falling. Epoch-to-epoch validation noise (±40%, with occasional
early spikes to 0.02–0.04) is larger than those late-stage gains, so which
epoch wins the best-checkpoint pick is partly luck. For this data and
architecture, 60–80 epochs would have been enough.

### On GPU choice

The bottleneck is not raw compute. The A1 network is narrow (16 and 8
channels) but 20 layers deep, so every layer is a small kernel that cannot
fill a modern GPU, and the layers are serially dependent. The run above
sustains on the order of 65 GFLOPS against the RX 6800's ~16 TFLOPS FP32
peak, well under 1% utilization, with the wall clock going to kernel launches
and memory round-trips rather than arithmetic.

This suggests, though it has not been measured, that a considerably weaker
discrete GPU would train at a broadly similar speed, with memory bandwidth
rather than FLOPS being the main differentiator. Memory capacity is not a
constraint either: activations only amount to a few hundred MB.

## Notes

- Runs on tinygrad's default device.
- Validation: `uv run tests/test_poc.py` checks receptive field, forward
  parity against an independent numpy implementation, `.nam` export
  round-trip, dataset alignment, and a training smoke test.
