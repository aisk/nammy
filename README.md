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
$ uv run main.py train input.wav output.wav --epochs 100 --out model.nam
```

`input.wav` is the DI/reamp source, `output.wav` the processed capture; both
must share a sample rate and be time-aligned (use `--latency <samples>` to
compensate reamp latency).

Run audio through a trained model (reamp):

```console
$ uv run main.py process model.nam input.wav output.wav
```

This also loads classic-schema (non-gated Tanh WaveNet) `.nam` files trained
elsewhere.

## Notes

- Runs on tinygrad's default device. Without a GPU and without `clang`
  installed, it falls back to the libLLVM CPU renderer automatically.
- Validation: `uv run tests/test_poc.py` checks receptive field, forward
  parity against an independent numpy implementation, `.nam` export
  round-trip, dataset alignment, and a training smoke test.
