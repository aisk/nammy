# nammy

A proof-of-concept trainer for [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)'s
**WaveNet architectures (A1 and A2)**, implemented with [tinygrad](https://github.com/tinygrad/tinygrad)
instead of PyTorch.

<img alt="GUI" src="https://github.com/user-attachments/assets/1a00f279-a167-438e-aca7-978419e1cf34" />


## What's implemented

- The current standard **A2** WaveNet (the default; `--arch a2`, or `a2-lite`
  for the 3-channel variant): one 23-layer array with LeakyReLU, restarting
  dilations, mixed kernel sizes (6 and 15) and a 16-tap head conv,
  `head_scale = 0.01`, receptive field 6347 — a faithful port of
  `nam.models.wavenet` with the reference trainer's current config.
- The classic **A1** WaveNet (`--arch a1`): two layer arrays (16 and 8
  channels), dilations 1–512, kernel size 3, Tanh, residual 1x1s, per-array
  1x1 head rechannel, `head_scale = 0.02`, receptive field 4093.
- NAM-style data pipeline: WAV loading (PCM 16/24/32 and IEEE float),
  latency compensation, and `(nx+ny-1, ny)` window slicing matching
  `nam.data.Dataset`.
- Training matching NAM's standard learning config: Adam(lr=0.004),
  per-epoch exponential LR decay (gamma 0.993), MSE loss, ESR validation with
  best-checkpoint restore, JIT-compiled train step and batched JIT inference.
- Export to `.nam`: A1 models write the classic v0.5.4 schema every plugin
  version can read; A2 models write the current v0.7.0 schema (needs a recent
  NAM plugin). Both schemas load back via `process`/the GUI.

Not implemented (yet): packed/slimmable training, gated/FiLM variants, MRSTFT
loss, pre-emphasis, output loudness normalization, the standardized input-file
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

## Notes

- Every command takes `--device`; see [Backend](#backend).
- On an RX 6800 a 100-epoch run over a 340 s capture pair takes about 13 min.
  [PERFORMANCE.md](PERFORMANCE.md) has the measured numbers for both
  architectures and what the time is spent on.
- Tests: `uv run tests/test_poc.py` checks receptive field, forward
  parity against an independent numpy implementation, `.nam` export
  round-trip, dataset alignment, a training smoke test, backend selection, and
  the progress/stop hooks the GUI drives training through.
