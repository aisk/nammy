# Design notes

Why a few things are built the way they are. None of it is needed to use
nammy; see the [README](README.md) for that, and [PERFORMANCE.md](PERFORMANCE.md)
for the measured numbers.

## Backend probing

Opening a device proves little, so each candidate in the default chain is
tested by compiling and running a small kernel on it. tinygrad's CPU device,
for one, needs clang and opens perfectly well without it, failing only when the
first kernel is compiled, which is why the `CPU:X86` renderer, which emits
machine code in-process and needs nothing installed, is in the chain behind it.

An explicit `--device` is never probed away: if it cannot run, that is an error
rather than a silent fall back to something an order of magnitude slower.

## Several GPUs

tinygrad's OpenCL backend only asks the first OpenCL platform for devices. An
Intel iGPU and an NVIDIA card are two platforms, so whichever driver the ICD
loader happens to list first is all tinygrad sees, and the card may be
unreachable. nammy therefore collects the GPUs of every platform, puts the ones
that do not share memory with the host first, and installs that list as
tinygrad's before any CL device is opened. Microsoft's OpenCLOn12 platform
lists the same adapters again through D3D12, so it is only used when no native
driver offers a GPU.

tinygrad's `DEV` reads `CL:1` as the CL device with a renderer called `1`, so
nammy sets an indexed target as a `Target` whose device name carries the index,
which tinygrad takes as `Device.DEFAULT` as is.

## The GUI's single worker thread

Work runs on one background thread, and it has to be exactly one: tinygrad
caches compiled kernels in sqlite, and that connection can only be used by the
thread that opened it.

The device picker is shared by both tabs for a similar reason. tinygrad's
device is process-wide, so there is nothing per-tab to choose.

## The single-file build

`dist/nammy.pyzw` is a [zipapp](https://docs.python.org/3/library/zipapp.html):
an ordinary zip holding nammy and tinygrad with a `__main__.py` at its root,
which Windows opens with `pythonw` on a double click. Everything is imported
from inside the archive, so nothing is unpacked and nothing is written
anywhere.

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
