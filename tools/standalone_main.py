"""
Entry point for the single-file build, stored as __main__.py inside it.

tools/build_standalone.py puts this at the root of the archive. Python runs it
when the archive is opened and puts the archive itself on sys.path, so nammy
and tinygrad are imported straight out of the file. Nothing is unpacked, and
nothing is written anywhere.

Windows runs .pyzw without a console, which is what a GUI wants but also means
a traceback has nowhere to go, so failures are reported in a message box.
"""

import sys
import traceback

MIN_PYTHON = (3, 13)  # @@MIN_PYTHON@@


def report(message):
    """Say something went wrong, given there is likely no console to say it in."""
    sys.stderr.write(message + "\n")
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("nammy", message)
        root.destroy()
    except Exception:
        pass


def main():
    if sys.version_info < MIN_PYTHON:
        wanted = ".".join(str(part) for part in MIN_PYTHON)
        report(
            f"nammy needs Python {wanted} or newer, but this file was opened with "
            f"{sys.version.split()[0]} from\n\n{sys.executable}\n\n"
            "Install a current Python, for example from the Microsoft Store."
        )
        raise SystemExit(1)

    try:
        from nammy.gui import main as gui
    except Exception:
        report(f"Could not start nammy with\n\n{sys.executable}\n\n{traceback.format_exc()}")
        raise SystemExit(1)
    gui()


if __name__ == "__main__":
    main()
