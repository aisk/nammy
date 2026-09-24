"""
Which tinygrad backend nammy runs on.

tinygrad picks a device itself, trying vendor backends (METAL, AMD, NV, CUDA,
QCOM) ahead of CL, but nammy is developed and measured against OpenCL, so it
walks its own chain: Metal on macOS, then OpenCL, then the CPU. An explicit
choice is never second-guessed. If it does not work that is an error, not a
quiet fall back to something an order of magnitude slower.

Opening a device says little about whether it works: tinygrad's CPU device
JIT-compiles with clang and opens fine without it, failing only later when the
first kernel is compiled. So a probe has to actually run something.

A machine can have more than one GPU, typically an integrated one next to a
discrete one, and tinygrad's OpenCL backend only looks at the first OpenCL
platform: with an Intel iGPU and an NVIDIA card, which are separate platforms,
the card may not be reachable at all. So nammy lists the GPUs of every platform
itself, discrete ones first, and hands tinygrad that list. "CL" is the first of
them, "CL:1" the second and so on.
"""

from __future__ import annotations

import ctypes
import functools
import os
import platform
import re
import shutil
import sys

_selected: str | None = None
# A threshold the user set for themselves outranks anything picked per device.
_USER_SPLIT_THRESHOLD = os.environ.get("REDUCEOP_SPLIT_THRESHOLD")


class DeviceError(RuntimeError):
    """A backend was asked for and cannot be used."""


def cpu_targets() -> list[str]:
    """
    CPU targets worth trying, best first.

    The default CPU renderer JIT-compiles with clang, so it is only worth an
    attempt when clang is installed. The x86 renderer emits machine code
    in-process and needs nothing; libLLVM is no more likely to be present than
    clang, so it serves only as the non-x86 fallback.
    """
    fallback = "CPU:X86" if platform.machine().lower() in ("amd64", "x86_64") else "CPU:LLVM"
    return (["CPU"] if shutil.which("clang") else []) + [fallback]


@functools.cache
def _cl_gpus() -> tuple[str, ...]:
    """
    Names of the OpenCL GPUs, in the order nammy numbers them, and tinygrad told
    to use that order.

    Discrete GPUs come first, told apart by not sharing memory with the host.
    Microsoft's OpenCLOn12 exposes every D3D12 adapter again as a translation
    layer, so its devices are only used when no native driver offers a GPU.
    Empty when there is no OpenCL library or platform at all, or when tinygrad
    has already opened a CL device and fixed its own list; either way tinygrad is
    left to itself.
    """
    try:
        from tinygrad.runtime.autogen import opencl as cl
        from tinygrad.runtime.ops_cl import CLDevice
    except Exception:
        return ()
    if CLDevice.device_ids is not None:
        return ()

    def text(get, handle, param) -> str:
        buf = ctypes.create_string_buffer(256)
        get(handle, param, len(buf), buf, None)
        return buf.value.decode(errors="replace").strip()

    count = ctypes.c_uint32()
    try:
        if cl.clGetPlatformIDs(0, None, ctypes.byref(count)) != 0 or not count.value:
            return ()
    except AttributeError:  # the binding loads its library lazily; there is none
        return ()
    platforms = (cl.cl_platform_id * count.value)()
    cl.clGetPlatformIDs(count.value, platforms, None)

    native, layered = [], []
    for plat in platforms:
        found = ctypes.c_uint32()
        if cl.clGetDeviceIDs(plat, cl.CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(found)) != 0:
            continue  # CL_DEVICE_NOT_FOUND: a CPU-only platform
        ids = (cl.cl_device_id * found.value)()
        cl.clGetDeviceIDs(plat, cl.CL_DEVICE_TYPE_GPU, found.value, ids, None)
        platform_name = text(cl.clGetPlatformInfo, plat, cl.CL_PLATFORM_NAME)
        (layered if "OpenCLOn12" in platform_name else native).extend(ids)
    gpus = native or layered
    if not gpus:
        return ()

    def integrated(dev) -> bool:
        flag = ctypes.c_uint32()
        cl.clGetDeviceInfo(dev, cl.CL_DEVICE_HOST_UNIFIED_MEMORY, 4, ctypes.byref(flag), None)
        return bool(flag.value)

    gpus.sort(key=integrated)  # stable, so each driver's own order survives
    CLDevice.device_ids = (cl.cl_device_id * len(gpus))(*gpus)
    return tuple(text(cl.clGetDeviceInfo, dev, cl.CL_DEVICE_NAME) for dev in gpus)


def cl_targets() -> list[str]:
    """One CL target per GPU, best first; just "CL" when there is at most one."""
    return ["CL"] + [f"CL:{i}" for i in range(1, len(_cl_gpus()))]


def describe(target: str) -> str | None:
    """The hardware behind a target, where nammy knows it (OpenCL GPUs only)."""
    base, index = _split_index(target)
    names = _cl_gpus()
    if base.upper() == "CL" and index < len(names):
        return names[index]
    return None


def preference() -> tuple[str, ...]:
    """The chain tried when no device is asked for, best first."""
    accelerators = (["METAL"] if sys.platform == "darwin" else []) + cl_targets()
    return tuple(accelerators + cpu_targets())


def _split_index(target: str) -> tuple[str, int]:
    """"CL:1" -> ("CL", 1); anything without a numeric suffix has index 0."""
    if m := re.fullmatch(r"([A-Za-z]+):(\d+)", target):
        return m[1], int(m[2])
    return target, 0


def _dev_value(target: str):
    """
    What DEV has to be set to for target.

    DEV reads "CL:1" as the CL device with a renderer called "1", so an indexed
    device has to be given as a Target whose device name carries the index; that
    becomes Device.DEFAULT as is, and tinygrad opens that device.
    """
    from tinygrad.helpers import Target

    base, index = _split_index(target)
    if base == target:
        return target
    return base.upper() if index == 0 else Target(device=f"{base.upper()}:{index}")


def probe(target: str) -> str | None:
    """Run a tiny kernel on target; None if it works, else why it does not."""
    from tinygrad import Tensor
    from tinygrad.device import ALL_DEVICES, Device
    from tinygrad.helpers import DEV

    # Before anything opens CL, or tinygrad keeps its own device list.
    gpus = _cl_gpus()
    base, index = _split_index(target)
    if base.upper() == "CL" and gpus and index >= len(gpus):
        return f"no such GPU; OpenCL has {len(gpus)}: " + ", ".join(cl_targets())
    previous = DEV.value
    try:
        DEV.value = _dev_value(target)
        Device[DEV.device]
        (Tensor([1.0, 2.0, 3.0]) * 2).tolist()
        return None
    except ModuleNotFoundError as exc:
        # No ops_<name> module: almost always a device name that does not exist.
        return f"unknown device ({exc}); tinygrad has {', '.join(ALL_DEVICES)}"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}".rstrip(": ")
    finally:
        DEV.value = previous


def select(requested: str | None = None) -> str:
    """
    Make a device current and return it.

    :param requested: a tinygrad target such as "CL", "METAL" or "CPU:X86".
        Anything tinygrad knows is accepted, not just the ones in preference().
        When it is None the preference chain is walked instead.
    :raises DeviceError: if the requested device does not work, or if nothing in
        the chain does.
    """
    if requested:
        if (error := probe(requested)) is not None:
            raise DeviceError(f"device {requested!r} is not usable: {error}")
        return _activate(requested)
    tried = []
    for target in preference():
        if (error := probe(target)) is None:
            return _activate(target)
        tried.append(f"{target} ({error})")
    raise DeviceError("no usable device found; tried " + ", ".join(tried))


def ensure_selected() -> str:
    """Select a device unless one already is, honouring DEV from the environment."""
    return _selected if _selected is not None else select(os.environ.get("DEV") or None)


def current() -> str | None:
    """The target selected so far, or None if the choice has not been made yet."""
    return _selected


def _activate(target: str) -> str:
    global _selected
    from tinygrad.helpers import DEV

    DEV.value = _dev_value(target)
    _tune_for(target)
    _selected = target
    return target


def _tune_for(target: str) -> None:
    """
    Splitting long reduces is a large win on a GPU and a small loss on a CPU (see
    the README), so it follows the device rather than being set once. tinygrad
    reads the threshold through a cached getenv when it schedules, so the cache
    has to go when the value changes.
    """
    if _USER_SPLIT_THRESHOLD is not None:
        return
    from tinygrad.helpers import getenv

    if target.split(":")[0].upper() == "CPU":
        os.environ.pop("REDUCEOP_SPLIT_THRESHOLD", None)
    else:
        os.environ["REDUCEOP_SPLIT_THRESHOLD"] = "8192"
    getenv.cache_clear()
