import os
import shutil
import sys

# tinygrad's default CPU renderer JIT-compiles with clang; fall back to the
# libLLVM renderer when clang isn't installed. Must run before tinygrad imports.
if "DEV" not in os.environ and "tinygrad" not in sys.modules and shutil.which("clang") is None:
    os.environ["DEV"] = "CPU:LLVM"

from .wavenet import WaveNet, a1_config

__all__ = ["WaveNet", "a1_config"]
