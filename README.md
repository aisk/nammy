# nammy

A proof-of-concept trainer for [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)'s
classic **A1 WaveNet architecture**, implemented with [tinygrad](https://github.com/tinygrad/tinygrad)
instead of PyTorch.

<img alt="GUI" src="https://github.com/user-attachments/assets/1a00f279-a167-438e-aca7-978419e1cf34" />


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
  best-checkpoint restore, JIT-compiled train step and batched JIT inference.
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

`--out` is rewritten every time validation ESR improves, so interrupting a run
leaves the best model so far on disk rather than nothing.

Run audio through a trained model (reamp):

```console
$ uv run python -m nammy process model.nam input.wav output.wav
```

This also loads classic-schema (non-gated Tanh WaveNet) `.nam` files trained
elsewhere.

### Backend

By default nammy tries Metal (on macOS), then OpenCL, then the CPU, and uses
the first that works. Pick one explicitly with `--device`, accepted by every
command:

```console
$ uv run python -m nammy train input.wav output.wav --device CL
```

Any tinygrad target is accepted, not just the ones in the default chain, so
`--device AMD`, `--device CUDA` or `--device CPU:X86` all work. An explicit
choice is never second-guessed: if it cannot run, that is an error rather than a
silent fall back to something an order of magnitude slower.

Opening a device proves little, so each candidate is tested by compiling and
running a small kernel on it. tinygrad's CPU device, for one, needs clang and
opens perfectly well without it, failing only when the first kernel is compiled
— which is why the `CPU:X86` renderer, which emits machine code in-process and
needs nothing installed, is in the chain behind it.

### GUI

There is a small Tkinter front end over the same two commands:

```console
$ uv run python -m nammy gui
```

On Windows you can instead double-click `nammy-gui.pyw`, which re-execs into
`.venv` so the system Python does not need the dependencies installed.

The Train tab streams the training log, plots validation ESR per epoch, and has
a Stop button that ends the run at the next batch boundary; because `--out` is
rewritten on every improvement, stopping leaves the best model so far on disk.
The Process tab reamps a WAV through a `.nam`. The device picker at the top is
shared by both tabs, since tinygrad's device is process-wide; it starts on the
best candidate that passed its probe, and picking one that failed says why.

Work runs on a single background thread, so the window stays responsive. It has
to be a single one: tinygrad caches compiled kernels in sqlite, and that
connection can only be used by the thread that opened it.

### Single-file build

For handing the GUI to someone who has a Python but no interest in installing
anything:

```console
$ uv run python tools/build_standalone.py     # -> dist/nammy.pyzw
```

That is a [zipapp](https://docs.python.org/3/library/zipapp.html): an ordinary
zip holding nammy and tinygrad with a `__main__.py` at its root, which Windows
opens with `pythonw` on a double click. Everything is imported from inside the
archive, so nothing is unpacked and nothing is written anywhere.

It works because neither package needs compiled code, which zipimport cannot
load. numpy is nammy's one compiled dependency and it is optional:
`nammy/_numpy_compat.py` stands in for the parts that get used when numpy is
missing, and `tests/test_no_numpy.py` runs the same work both ways and compares
the results.

tinygrad needs two patches to run from an archive, both applied by the build
and both a consequence of there being no real directory to look at: it reads
the backend list by listing `runtime/`, and it regenerates its ctypes bindings
over the network when it cannot find their `.py` file. Both patches are matched
exactly, so a tinygrad upgrade that moves the ground under them fails the build
with a message rather than quietly producing a file that does not work.

Windows Defender's Controlled Folder Access may report that it "blocked
python.exe from making changes to memory" (event 1127). That is tinygrad's JIT
allocating executable memory, it happens however nammy is started, and it has
not stopped a run here on either the OpenCL or the CPU backend. Allowing the
interpreter under Ransomware protection silences it.

## Example training run

Reference numbers from a run on consumer hardware, training on the Blackstar
HT-1 capture pair from [Alec Wright's dataset](https://github.com/Alec-Wright/Automated-GuitarAmpModelling)
(340 s of aligned input/target at 44.1 kHz):

- **Hardware**: AMD Radeon RX 6800 on Windows 11, via tinygrad's OpenCL (`CL`)
  backend.
- **Config**: defaults — 100 epochs, batch 16, `ny` 8192, Adam(lr=0.004).
- **Speed**: 61 s for the first epoch (dominated by JIT compilation), then a
  steady ~8 s per epoch; 14 min total.
- **Result**: best validation ESR **0.0070**, reached at epoch 88. For
  reference, ESR below 0.01 is a good model and 0.02–0.05 is usable.

Convergence is fast: ESR hits 0.054 by epoch 3 and 0.019 by epoch 10, then
flattens while the training loss keeps falling. The best checkpoint stands at
0.0086 by epoch 60 and 0.0079 by epoch 80, so the last 40 epochs are still worth
about 20%. Epoch-to-epoch validation noise is larger than that, though: ESR
bounces between 0.0070 and 0.018 across epochs 40–100, so which epoch wins the
best-checkpoint pick is partly luck. At 8 s per epoch there is little reason to
stop early.

### Where the time goes

The bottleneck is not raw compute. The A1 network is narrow (16 and 8
channels) but 20 layers deep, so every layer is a small kernel that cannot
fill a modern GPU, and the layers are serially dependent. A training step at
the default batch 16 / `ny` 8192 issues around 800 kernels and sustains
~350 GFLOPS by tinygrad's op counter, against the RX 6800's ~16 TFLOPS FP32
peak, a few percent of the card.

Reaching even that took two fixes, both worth knowing about if you port this to
another backend:

- The backward pass reduces weight gradients over the (batch × time) axis,
  ~12k long against a handful of output elements. tinygrad only splits such a
  reduce across two kernels when the input/output element ratio reaches 32768,
  and this model sits just under that, so those reduces landed in single
  low-occupancy kernels: two of them alone cost 73 of the 188 ms step, running
  at 3–11 GFLOPS. `nammy/device.py` lowers `REDUCEOP_SPLIT_THRESHOLD` to 8192
  when it selects an accelerator, which takes the step to 63 ms; set it in the
  environment to override. A CPU has no occupancy to win back and measures a few
  percent slower, so the setting follows the device rather than being global.
- Validation ran one 65536-sample chunk at a time at batch 1, rebuilding the
  graph in Python for each. Chunks are independent, so `WaveNet.process` stacks
  them on the batch axis under a JIT: 5.3 s → 0.1 s per epoch.

Together those took the epoch from ~26 s to ~8 s, with the same results to
within float reordering. Memory is not a constraint: activations peak around
1 GB at the default batch and `ny`.

## Notes

- Every command takes `--device`; see [Backend](#backend).
- Tests: `uv run tests/test_poc.py` checks receptive field, forward
  parity against an independent numpy implementation, `.nam` export
  round-trip, dataset alignment, a training smoke test, backend selection, and
  the progress/stop hooks the GUI drives training through.
