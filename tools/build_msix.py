"""
Build nammy into an .msix for the Microsoft Store.

    python tools/build_msix.py                     # -> dist/nammy-0.3.0-x64.msix
    python tools/build_msix.py --self-sign         # ... and sign it, to install locally
    python tools/build_msix.py --identity-name 1234Publisher.nammy --publisher "CN=..."

Four steps: draw the icons, freeze the GUI with PyInstaller, repair the payload,
then pack it with the manifest into an MSIX. See PACKAGING.md for what to do
with the file afterwards, and for where the Identity values come from.

Packages submitted to the Store are re-signed with the Store's certificate, so
the one produced by default is unsigned and Windows will not install it.
--self-sign issues a certificate matching the manifest's Publisher and signs
with it, which is the only way to run the packaged app before submitting;
-AllowUnsigned is not a way round it, as it refuses any package that declares an
executable to launch.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import struct
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ElementTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"
BUILD = ROOT / "build"
LAYOUT = BUILD / "msix"
FROZEN = BUILD / "pyinstaller"
CERTIFICATE = BUILD / "nammy-selfsigned.pfx"
CERTIFICATE_PASSWORD = "nammy"  # a local test certificate, not a secret

APPX = "http://schemas.microsoft.com/appx/manifest/foundation/windows10"
# Every prefix the manifest uses has to be registered before it is written back
# out, or ElementTree renames them to ns0 and ns1 and the IgnorableNamespaces
# attribute, which is plain text naming the prefixes, ends up pointing at
# nothing. makeappx rejects that with "an XML prefix that is not defined".
NAMESPACES = {
    "": APPX,
    "uap": "http://schemas.microsoft.com/appx/manifest/uap/windows10",
    "rescap": "http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities",
}


def version() -> str:
    """pyproject's version as MSIX wants it: four parts, the last one zero.

    The Store rejects a package whose revision is not 0, reserving it for its own
    republishing, so there are only three parts to carry over.
    """
    with open(ROOT / "pyproject.toml", "rb") as fp:
        declared = tomllib.load(fp)["project"]["version"]
    parts = declared.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise SystemExit(f"version {declared!r} is not three numbers; MSIX needs x.y.z")
    return ".".join(parts + ["0"])


def sdk_tool(name: str) -> pathlib.Path:
    """The newest x64 build of a Windows SDK tool, which is not on PATH by default."""
    roots = [
        pathlib.Path(base) / "Windows Kits" / "10" / "bin"
        for base in ("C:/Program Files (x86)", "C:/Program Files")
    ]
    found = sorted(
        (path for root in roots if root.is_dir() for path in root.glob(f"*/x64/{name}")),
        key=lambda path: path.parent.parent.name,
    )
    if not found:
        raise SystemExit(
            f"cannot find {name}; install the Windows SDK "
            "(the 'Windows SDK Signing Tools' component is enough)"
        )
    return found[-1]


def run(command: list[str | pathlib.Path], what: str) -> None:
    print(f"  $ {' '.join(str(part) for part in command)}")
    if subprocess.run(command).returncode != 0:
        raise SystemExit(f"{what} failed")


def freeze() -> pathlib.Path:
    """Run PyInstaller over tools/nammy.spec and return the collected directory."""
    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--log-level",
            "WARN",
            "--distpath",
            FROZEN,
            "--workpath",
            BUILD / "pyinstaller-work",
            ROOT / "tools" / "nammy.spec",
        ],
        "PyInstaller",
    )
    collected = FROZEN / "nammy"
    if not (collected / "nammy.exe").is_file():
        raise SystemExit(f"PyInstaller produced no exe in {collected}")
    return collected


def manifest(identity_name: str | None, publisher: str | None) -> str:
    """packaging/AppxManifest.xml with Version, and any overridden Identity, stamped in."""
    for prefix, uri in NAMESPACES.items():
        ElementTree.register_namespace(prefix, uri)
    tree = ElementTree.parse(PACKAGING / "AppxManifest.xml")
    identity = tree.getroot().find(f"{{{APPX}}}Identity")
    identity.set("Version", version())
    if identity_name:
        identity.set("Name", identity_name)
    if publisher:
        identity.set("Publisher", publisher)
    print(f"  identity {identity.get('Name')} {identity.get('Version')} {identity.get('Publisher')}")
    return ElementTree.tostring(tree.getroot(), encoding="unicode", xml_declaration=True)


def lay_out(collected: pathlib.Path, identity_name: str | None, publisher: str | None) -> None:
    """
    Assemble what makeappx packs: the frozen app, the tiles, and the manifest.

    Rebuilt from scratch every time. A stale file left behind in a package layout
    ships, and the only sign of it is the package being larger than it should be.
    """
    if LAYOUT.exists():
        shutil.rmtree(LAYOUT)
    shutil.copytree(collected, LAYOUT / "app")
    shutil.copytree(PACKAGING / "assets", LAYOUT / "Assets")
    (LAYOUT / "AppxManifest.xml").write_text(manifest(identity_name, publisher), encoding="utf-8")


def clear_dangling_signatures(root: pathlib.Path) -> list[str]:
    """
    Zero the certificate table of any binary whose signature was stripped badly.

    tcl86t.dll and tk86t.dll as shipped by python-build-standalone, which is
    where uv's interpreters come from, have had their Authenticode signature
    removed without the certificate table entry in the optional header being
    cleared with it, so the header still points at 7408 bytes starting exactly
    at the end of the file. Nothing notices until signtool walks the binaries in
    a package, refuses the malformed one, and fails the whole thing with
    0x800700C1, ERROR_BAD_EXE_FORMAT, naming no file.

    Clearing the entry costs nothing: it says there is no signature, which is
    true, and it is what stripping one is supposed to leave behind.
    """
    repaired = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in (".dll", ".pyd", ".exe") or not path.is_file():
            continue
        data = bytearray(path.read_bytes())
        if data[:2] != b"MZ":
            continue
        header = struct.unpack_from("<I", data, 0x3C)[0]
        if data[header : header + 4] != b"PE\0\0":
            continue
        # Certificate table: the fifth data directory, which follows the optional
        # header, whose length depends on whether the binary is PE32 or PE32+.
        magic = struct.unpack_from("<H", data, header + 24)[0]
        entry = header + 24 + (112 if magic == 0x20B else 96) + 8 * 4
        offset, size = struct.unpack_from("<II", data, entry)
        if offset and offset + size > len(data):
            struct.pack_into("<II", data, entry, 0, 0)
            path.write_bytes(data)
            repaired.append(path.name)
    return repaired


def pack(output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    run([sdk_tool("makeappx.exe"), "pack", "/d", LAYOUT, "/p", output, "/o"], "makeappx")


def self_sign(output: pathlib.Path) -> None:
    """
    Sign with a throwaway certificate whose subject matches the manifest Publisher.

    Windows will not install a package whose signing certificate says anything
    other than exactly what Identity/@Publisher says, so the two are read from
    the same place rather than both being typed in.
    """
    root = ElementTree.parse(LAYOUT / "AppxManifest.xml").getroot()
    publisher = root.find(f"{{{APPX}}}Identity").get("Publisher")
    if not CERTIFICATE.exists():
        print(f"  issuing {CERTIFICATE.name} for {publisher}")
        run(
            [
                # PowerShell 7 where it exists: Windows PowerShell has been seen
                # failing to autoload the PKI module, and then Cert:\ is not a
                # drive and New-SelfSignedCertificate is not a command.
                shutil.which("pwsh") or "powershell.exe",
                "-NoProfile",
                "-Command",
                f"$c = New-SelfSignedCertificate -Type CodeSigningCert "
                f"-Subject '{publisher}' -CertStoreLocation Cert:\\CurrentUser\\My "
                f"-KeyUsage DigitalSignature -FriendlyName 'nammy test signing' "
                f"-TextExtension @('2.5.29.37={{text}}1.3.6.1.5.5.7.3.3'); "
                f"Export-PfxCertificate -Cert $c -FilePath '{CERTIFICATE}' "
                f"-Password (ConvertTo-SecureString -String '{CERTIFICATE_PASSWORD}' "
                f"-Force -AsPlainText) | Out-Null",
            ],
            "issuing a test certificate",
        )
    run(
        [
            sdk_tool("signtool.exe"),
            "sign",
            "/fd",
            "SHA256",
            "/f",
            CERTIFICATE,
            "/p",
            CERTIFICATE_PASSWORD,
            output,
        ],
        "signtool",
    )
    print(
        f"\nsigned with a self-issued certificate. To install it, trust that certificate\n"
        f"once from an elevated prompt, then add the package:\n"
        f"  certutil -addstore TrustedPeople {CERTIFICATE}\n"
        f"  Add-AppxPackage {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, help="defaults to dist/nammy-<version>-x64.msix"
    )
    parser.add_argument("--identity-name", help="Identity/@Name, as reserved in Partner Center")
    parser.add_argument("--publisher", help="Identity/@Publisher, as assigned in Partner Center")
    parser.add_argument(
        "--self-sign", action="store_true", help="sign it, so the package can be installed locally"
    )
    parser.add_argument(
        "--skip-assets", action="store_true", help="reuse the icons already in packaging/"
    )
    args = parser.parse_args()

    output = args.output or ROOT / "dist" / f"nammy-{version().rsplit('.', 1)[0]}-x64.msix"

    if not args.skip_assets:
        print("icons")
        run([sys.executable, ROOT / "tools" / "make_assets.py"], "drawing the icons")
    print("freezing")
    collected = freeze()
    print("layout")
    lay_out(collected, args.identity_name, args.publisher)
    repaired = clear_dangling_signatures(LAYOUT / "app")
    print(f"  cleared a dangling certificate table in {', '.join(repaired) or 'nothing'}")
    print("packing")
    pack(output)
    if args.self_sign:
        print("signing")
        self_sign(output)

    size = output.stat().st_size / 1e6
    payload = sum(path.stat().st_size for path in LAYOUT.rglob("*") if path.is_file()) / 1e6
    print(f"\n{output}\n  {size:.1f} MB packed, {payload:.1f} MB laid out")


if __name__ == "__main__":
    sys.exit(main())
