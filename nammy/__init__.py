import os
import shutil
import sys

# tinygrad's default CPU renderer JIT-compiles with clang; fall back to the
# libLLVM renderer when clang isn't installed. Must run before tinygrad imports.
if "DEV" not in os.environ and "tinygrad" not in sys.modules and shutil.which("clang") is None:
    os.environ["DEV"] = "CPU:LLVM"

# The backward pass reduces weight gradients over the (batch x time) axis, which
# is ~12k long here while the output is a handful of elements. tinygrad only
# splits such a reduce across two kernels when the input/output element ratio
# reaches 32768, and this model sits just under that, so the reduce lands in one
# low-occupancy kernel: two of them alone cost 73 of the 188 ms per step. Lowering
# the threshold triples step throughput (188 -> 63 ms, 118 -> 348 GFLOPS on an
# RX 6800). Also read before tinygrad imports (helpers.getenv is cached).
os.environ.setdefault("REDUCEOP_SPLIT_THRESHOLD", "8192")

from .wavenet import WaveNet, a1_config

__all__ = ["WaveNet", "a1_config"]
