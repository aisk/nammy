# Packaging for the Microsoft Store

nammy ships to the Store as an MSIX holding a frozen copy of the GUI. Nothing
here is needed to use nammy; see the [README](README.md) for that.

```console
$ uv run --group packaging python tools/build_msix.py
```

That draws the icons, freezes the GUI with PyInstaller, assembles a package
layout under `build/msix`, and packs it to `dist/nammy-<version>-x64.msix`,
about 17 MB. The steps live in `tools/build_msix.py`, the freeze is configured
in `tools/nammy.spec`, and the manifest is `packaging/AppxManifest.xml`.

## Why the interpreter ships too

An MSIX cannot depend on anything being installed and its install directory is
read-only, so there is no version of this that finds a Python on the machine.
The package therefore carries one, along with tkinter's Tcl/Tk, tinygrad, and
nammy itself. numpy is deliberately left out: `nammy/_numpy_compat.py` covers
what nammy and tinygrad ask of it, and without it the payload is pure Python
with no extension modules of its own and no MSVC runtime to redistribute.
`tools/nammy.spec` says how to put it back if WAV loading turns out to want it.

tinygrad is collected as plain `.py` files rather than folded into PyInstaller's
archive, because it finds its backends by listing its own `runtime/` directory.
That is the same thing `tools/build_standalone.py` has to patch around for the
zipapp; here it just needs the files to be real.

The app runs as `Windows.FullTrustApplication`, which is an ordinary Win32
process with an MSIX identity rather than a sandboxed one. tinygrad's JIT
allocates executable memory and its GPU backends load the vendor OpenCL ICD out
of the system, neither of which a sandboxed package may do. This is what the
`runFullTrust` capability declares, and the submission has to say why the app
needs it: it compiles and runs GPU kernels at runtime.

## Running the package before submitting

An unsigned MSIX will not install. `-AllowUnsigned` is not a way round it either:
it refuses any package declaring an executable to launch, which is every desktop
app. So sign it with a throwaway certificate:

```console
$ uv run --group packaging python tools/build_msix.py --self-sign
```

That issues one matching the manifest's `Publisher` into `build/`, and signs.
Installing it means trusting that certificate once, which needs an elevated
prompt:

```console
> certutil -addstore TrustedPeople build\nammy-selfsigned.pfx
> Add-AppxPackage dist\nammy-0.3.0-x64.msix
```

`Remove-AppxPackage AnLong.nammy_0.3.0.0_x64__<hash>` takes it away again. None
of this touches the Store build, which Microsoft re-signs.

## Submitting

1. Reserve the name in [Partner Center](https://partner.microsoft.com/dashboard).
   An individual account is a one-time 19 USD; no code signing certificate is
   needed, and that is most of the reason to prefer MSIX over shipping an
   installer yourself.
2. Read the reserved app's **Product identity** page. It gives a package
   `Identity/@Name` (something like `12345AnLong.nammy`) and a `Publisher`
   (`CN=` followed by a GUID). Build with both, since a package whose identity
   does not match the reservation is rejected on upload:

   ```console
   $ uv run --group packaging python tools/build_msix.py \
       --identity-name 12345AnLong.nammy --publisher "CN=<the GUID>"
   ```

3. Upload the `.msix` to the submission's Packages page. Do not sign it; the
   Store re-signs with its own certificate, which is also why the version's
   fourth part has to stay 0.
4. The rest of the submission: screenshots, a description, the age rating
   questionnaire, and a privacy policy URL. nammy makes no network calls and
   collects nothing, but the field still wants an address.
5. Put a link to an input/output WAV pair in **Notes for certification**.
   Without material to train on, a reviewer cannot get past the first screen.

## Known rough edges

- The backends are not verified from inside a package. tinygrad picks OpenCL
  when a driver is present and otherwise falls back to `CPU:X86`, which emits
  machine code in process; `shutil.which("clang")` finds nothing in a package
  so the clang-backed CPU target is skipped on its own. Both paths are worth
  running once on a machine without a discrete GPU before submitting.
- Windows Defender's Controlled Folder Access may still report event 1127
  against the app, as the README describes for the unpackaged build. Signing
  does not change it: it is the JIT allocating executable memory.
- The tiles are unqualified filenames with no `resources.pri` beside them, so
  Windows rescales one image per tile rather than picking a per-scale asset. The
  certification kit may note it. `makepri` from the same SDK is the fix if it
  ever matters.
- `tools/build_msix.py` clears a certificate table left dangling in `tcl86t.dll`
  and `tk86t.dll`. Those ship that way from python-build-standalone, which is
  where uv's interpreters come from, and signtool rejects a package containing
  them with `0x800700C1`, naming no file. The comment on
  `clear_dangling_signatures` has the detail.
