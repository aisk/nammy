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
"""

from __future__ import annotations

import os
import platform
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


def preference() -> tuple[str, ...]:
    """The chain tried when no device is asked for, best first."""
    accelerators = ["METAL", "CL"] if sys.platform == "darwin" else ["CL"]
    return tuple(accelerators + cpu_targets())


def probe(target: str) -> str | None:
    """Run a tiny kernel on target; None if it works, else why it does not."""
    from tinygrad import Tensor
    from tinygrad.device import ALL_DEVICES, Device
    from tinygrad.helpers import DEV

    previous = DEV.value
    try:
        DEV.value = target
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

    DEV.value = target
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
