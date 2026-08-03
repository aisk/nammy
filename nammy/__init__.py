import os
import platform
import shutil
import sys

# tinygrad's default CPU renderer JIT-compiles with clang. Without it, fall back
# to the x86 renderer, which emits machine code in-process and needs nothing
# installed; the libLLVM renderer is no help here since libLLVM is no more likely
# to be present than clang. Must run before tinygrad imports.
if "DEV" not in os.environ and "tinygrad" not in sys.modules and shutil.which("clang") is None:
    x86 = platform.machine().lower() in ("amd64", "x86_64")
    os.environ["DEV"] = "CPU:X86" if x86 else "CPU:LLVM"

# The backward pass reduces weight gradients over the (batch x time) axis, which
# is ~12k long here while the output is a handful of elements. tinygrad only
# splits such a reduce across two kernels when the input/output element ratio
# reaches 32768, and this model sits just under that, so the reduce lands in one
# low-occupancy kernel: two of them alone cost 73 of the 188 ms per step. Lowering
# the threshold triples step throughput (188 -> 63 ms, 118 -> 348 GFLOPS on an
# RX 6800). Also read before tinygrad imports (helpers.getenv is cached).
#
# A CPU has no occupancy to win back and pays for the extra round trip, measuring
# a few percent slower, so skip it there. This only recognizes an explicit DEV:
# with none set, tinygrad picks the device itself and prefers an accelerator, and
# asking it here would mean initializing a device at import time.
if not os.environ.get("DEV", "").upper().startswith("CPU"):
    os.environ.setdefault("REDUCEOP_SPLIT_THRESHOLD", "8192")

from .wavenet import WaveNet, a1_config

__all__ = ["WaveNet", "a1_config"]
