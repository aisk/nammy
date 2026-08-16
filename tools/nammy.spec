# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the packaged Windows GUI. Driven by tools/build_msix.py.

Two settings here carry the whole build:

`module_collection_mode` keeps tinygrad as real .py files under _internal/
instead of folding it into the PYZ archive. tinygrad finds its backends by
listing its own runtime/ directory and regenerates ctypes bindings when their
file is missing, neither of which survives being an archive member; on disk both
work untouched. tools/build_standalone.py patches around the same two spots
because a zipapp has no other option, and this build wants none of that.

numpy is excluded on purpose. nammy falls back to nammy/_numpy_compat.py, which
implements the slice of numpy it and tinygrad reach for, so leaving numpy out
makes the whole package pure Python: no extension modules to collect, no MSVC
runtime to redistribute, nothing that can fail to load out of an install
directory it cannot write to. Drop the exclusion if WAV loading measures too
slow through the stand-in; nothing else depends on the choice.
"""

import pathlib

from PyInstaller.utils.hooks import collect_submodules

ROOT = pathlib.Path(SPECPATH).parent

analysis = Analysis(
    [str(ROOT / "tools" / "frozen_main.py")],
    pathex=[str(ROOT)],
    # tinygrad imports its backends by name at runtime, so nothing static points
    # at ops_*.py and the analysis would collect none of them.
    hiddenimports=collect_submodules("tinygrad"),
    excludes=["numpy", "torch", "pydantic", "pytest"],
    module_collection_mode={"tinygrad": "py"},
    noarchive=False,
)

exe = EXE(
    PYZ(analysis.pure, analysis.zipped_data),
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="nammy",
    icon=str(ROOT / "packaging" / "nammy.ico"),
    console=False,  # the GUI shows its own log; a console here would be a stray window
    debug=False,
    strip=False,
    upx=False,  # compressed sections are what heuristic scanners object to
)

COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    name="nammy",
    strip=False,
    upx=False,
)
