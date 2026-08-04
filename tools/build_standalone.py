"""
Build nammy into one double-clickable .pyzw.

The result is a zipapp: a plain zip holding nammy and tinygrad with a
`__main__.py` at its root, which Python runs when the file is opened. Windows
associates .pyzw with pythonw, so it starts on a double click, and everything
is imported from inside the archive with nothing unpacked to disk.

    python tools/build_standalone.py            # -> dist/nammy.pyzw

This works only because neither package needs compiled code, which zipimport
cannot load: numpy is nammy's one compiled dependency and it is optional, with
nammy/_numpy_compat.py standing in for it.

tinygrad does need two patches to run from an archive, both applied here and
both a consequence of there being no real directory to look at. They are
matched exactly, so a tinygrad upgrade that moves the ground under them fails
the build rather than quietly producing a file that does not work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import pathlib
import sys
import zipfile
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTRY_POINT = pathlib.Path(__file__).with_name("standalone_main.py")
# Everything nammy imports that is not in the standard library, all of which
# has to be pure Python.
DEPENDENCIES = ("tinygrad",)


@dataclass
class Patch:
    """A single exact edit to a bundled file, applied on the way into the zip."""

    path: str
    old: str
    new: str
    why: str
    applied: int = 0


def patches(tinygrad: pathlib.Path) -> list[Patch]:
    backends = sorted(p.stem[len("ops_") :].upper() for p in (tinygrad / "runtime").glob("ops_*.py"))
    return [
        Patch(
            "tinygrad/device.py",
            '[x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime")'
            '.iterdir() if x.stem.startswith("ops_")]',
            repr(backends),
            "the backend list is read by listing a directory, which an archive is not",
        ),
        Patch(
            "tinygrad/runtime/autogen/__init__.py",
            "def load(name, files, **kwargs):\n  if not (f:=",
            "def load(name, files, **kwargs):\n"
            "  # Bundled bindings are never on disk, and regenerating one downloads\n"
            "  # headers and runs a code generator, so try importing it first.\n"
            "  try: return importlib.import_module("
            "f\"{kwargs.get('path', __name__)}.{name.replace('/', '.')}\")\n"
            "  except ImportError: pass\n"
            "  if not (f:=",
            "ctypes bindings are regenerated over the network when their file is missing",
        ),
    ]


def source_files(package: pathlib.Path):
    """The .py files of a package, without the bytecode cache."""
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def locate(name: str) -> pathlib.Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"cannot find the {name} package; install it and try again")
    return pathlib.Path(list(spec.submodule_search_locations)[0])


def minimum_python() -> tuple[int, ...]:
    """The floor pyproject.toml declares, so the two cannot drift apart."""
    import tomllib

    with open(ROOT / "pyproject.toml", "rb") as fp:
        requires = tomllib.load(fp)["project"]["requires-python"]
    return tuple(int(part) for part in requires.lstrip("<>=~^ ").split("."))


def entry_point() -> str:
    """The archive's __main__.py, with the Python floor filled in."""
    lines = ENTRY_POINT.read_text(encoding="utf-8").splitlines(keepends=True)
    out = [
        f"MIN_PYTHON = {minimum_python()}  # @@MIN_PYTHON@@\n"
        if line.rstrip().endswith("# @@MIN_PYTHON@@")
        else line
        for line in lines
    ]
    return "".join(out)


def write_archive(path: pathlib.Path, packages: list[pathlib.Path], edits: list[Patch]) -> int:
    """
    Write the zipapp. Timestamps are fixed so the build is reproducible.

    No shebang is written. It would only be read on Unix, and on Windows there
    is a risk the launcher honours it and opens a console for a GUI program.
    """
    count = 0
    by_path: dict[str, list[Patch]] = {}
    for patch in edits:
        by_path.setdefault(patch.path, []).append(patch)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for package in packages:
            for source in source_files(package):
                name = source.relative_to(package.parent).as_posix()
                text = source.read_text(encoding="utf-8")
                for patch in by_path.get(name, ()):
                    patch.applied += text.count(patch.old)
                    text = text.replace(patch.old, patch.new)
                store(archive, name, text)
                count += 1
        store(archive, "__main__.py", entry_point())
    return count + 1


def store(archive: zipfile.ZipFile, name: str, text: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, text)


def check(edits: list[Patch]) -> None:
    """Every patch has to have landed exactly once, or the build is not trustworthy."""
    for patch in edits:
        if patch.applied != 1:
            raise SystemExit(
                f"patch for {patch.path} applied {patch.applied} times, expected 1.\n"
                f"  it exists because {patch.why}\n"
                f"  looked for: {patch.old!r}\n"
                "  the bundled package has changed; re-check the patch against it."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "-o", "--output", default=ROOT / "dist" / "nammy.pyzw", type=pathlib.Path
    )
    args = parser.parse_args()

    tinygrad = locate("tinygrad")
    packages = [ROOT / "nammy", tinygrad]
    edits = patches(tinygrad)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = write_archive(args.output, packages, edits)
    check(edits)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()[:12]
    print(f"{args.output}")
    print(f"  {files} files from {', '.join(p.name for p in packages)}")
    print(f"  {args.output.stat().st_size / 1e6:.2f} MB, sha256 {digest}")
    print(f"  {len(edits)} patches applied, needs Python {'.'.join(map(str, minimum_python()))}+")


if __name__ == "__main__":
    sys.exit(main())
