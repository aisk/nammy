# nammy

A proof-of-concept trainer for [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)'s
**WaveNet architectures (A1 and A2)**, implemented with [tinygrad](https://github.com/tinygrad/tinygrad)
instead of PyTorch.

<img alt="GUI" src="https://github.com/user-attachments/assets/1a00f279-a167-438e-aca7-978419e1cf34" />


## What's implemented

Both WaveNet architectures, ported faithfully from `nam.models.wavenet` with
the reference trainer's config: the current standard **A2** (the default;
`--arch a2`, or `a2-lite` for the 3-channel variant) and the classic **A1**
(`--arch a1`). Around them, NAM's data pipeline (WAV loading, latency
compensation, `nam.data.Dataset` window slicing) and its standard learning
config (Adam, per-epoch exponential LR decay, MSE loss, ESR validation with
best-checkpoint restore). Export to `.nam` works for both: A1 writes the
classic v0.5.4 schema every plugin version can read, A2 the current v0.7.0
schema, which needs a recent NAM plugin.

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
the first that actually compiles and runs a kernel. Pick one explicitly with
`--device`, accepted by every command:

```console
$ uv run python -m nammy train input.wav output.wav --device CL
```

Any tinygrad target is accepted, not just the ones in the default chain, so
`--device AMD`, `--device CUDA` or `--device CPU:X86` all work. An explicit
choice is never second-guessed: if it cannot run, that is an error rather than a
silent fall back to something an order of magnitude slower.

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
shared by both tabs; it starts on the best candidate that passed its probe, and
picking one that failed says why. Work runs on a background thread, so the
window stays responsive.

### Single-file build

For handing the GUI to someone who has a Python but no interest in installing
anything:

```console
$ uv run python tools/build_standalone.py     # -> dist/nammy.pyzw
```

That is a [zipapp](https://docs.python.org/3/library/zipapp.html): an ordinary
zip holding nammy and tinygrad with a `__main__.py` at its root, which Windows
opens with `pythonw` on a double click. Everything is imported from inside the
archive, so nothing is unpacked and nothing is written anywhere. It needs no
dependencies beyond a Python, not even numpy.

Windows Defender's Controlled Folder Access may report that it "blocked
python.exe from making changes to memory" (event 1127). That is tinygrad's JIT
allocating executable memory, it happens however nammy is started, and it has
not stopped a run here on either the OpenCL or the CPU backend. Allowing the
interpreter under Ransomware protection silences it.

## Notes

- On an RX 6800 a 100-epoch run over a 340 s capture pair takes about 13 min.
  [PERFORMANCE.md](PERFORMANCE.md) has the measured numbers for both
  architectures and what the time is spent on.
- [DESIGN.md](DESIGN.md) covers why the backend is probed, why the GUI uses a
  single worker thread, and how the zipapp build works.
- Tests: `uv run tests/test_poc.py` checks receptive field, forward
  parity against an independent numpy implementation, `.nam` export
  round-trip, dataset alignment, a training smoke test, backend selection, and
  the progress/stop hooks the GUI drives training through.
