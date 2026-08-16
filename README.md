# nammy

A proof-of-concept trainer for [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)'s
**WaveNet architectures (A1 and A2)**, implemented with [tinygrad](https://github.com/tinygrad/tinygrad)
instead of PyTorch.

<img alt="GUI" src="https://github.com/user-attachments/assets/1a00f279-a167-438e-aca7-978419e1cf34" />


## What's implemented

Both WaveNet architectures, ported faithfully from `nam.models.wavenet`: the
current standard **A2** (the default; `--arch a2`, or `a2-lite` for the
3-channel variant) and the classic **A1** (`--arch a1`), with NAM's data
pipeline and standard learning config around them. Trained models export to
`.nam` and load in the plugin, A2 in a recent version of it.

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
compensate reamp latency). `--out` is rewritten every time validation ESR
improves, so interrupting a run leaves the best model so far on disk.

Run audio through a trained model (reamp):

```console
$ uv run python -m nammy process model.nam input.wav output.wav
```

This also loads WaveNet `.nam` files trained elsewhere.

### Backend

By default nammy tries Metal (on macOS), then OpenCL, then the CPU, and uses
the first that actually compiles and runs a kernel. Any tinygrad target can be
chosen explicitly with `--device`, accepted by every command:

```console
$ uv run python -m nammy train input.wav output.wav --device CL
```

### GUI

A small Tkinter front end over the same two commands:

```console
$ uv run python -m nammy gui
```

On Windows you can instead double-click `nammy-gui.pyw`. The Train tab streams
the log, plots validation ESR per epoch, and can stop a run at the next batch
boundary; the Process tab reamps a WAV through a `.nam`. The device picker at
the top is shared by both tabs.

### Single-file build

For handing the GUI to someone who has a Python but no interest in installing
anything:

```console
$ uv run python tools/build_standalone.py     # -> dist/nammy.pyzw
```

That is a [zipapp](https://docs.python.org/3/library/zipapp.html): one file
holding nammy and tinygrad, double-clickable on Windows, needing nothing
installed and unpacking nothing.

Windows Defender's Controlled Folder Access may report that it "blocked
python.exe from making changes to memory" (event 1127). That is tinygrad's JIT
allocating executable memory and it has not stopped a run here; allowing the
interpreter under Ransomware protection silences it.

## Notes

- On an RX 6800 a 100-epoch run over a 340 s capture pair takes about 13 min.
  [PERFORMANCE.md](PERFORMANCE.md) has the numbers and where the time goes.
- [DESIGN.md](DESIGN.md) covers why the backend is probed, why the GUI uses a
  single worker thread, and how the zipapp build works.
- [PACKAGING.md](PACKAGING.md) covers the Microsoft Store build, which is an
  MSIX with an interpreter inside it rather than a zipapp.
- Tests: `uv run tests/test_poc.py` and `uv run tests/test_no_numpy.py`.
