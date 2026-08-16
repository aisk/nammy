"""
Entry point for the frozen Windows build, the script PyInstaller compiles in.

nammy-gui.pyw cannot serve here. It re-execs into the project virtualenv, which
is exactly wrong for a build that ships its own interpreter and has no .venv
beside it, and it checks the Python version, which is now settled at build time.
What is left is the part that still matters: a windowed build has no console, so
sys.stderr goes nowhere and a failure to import would otherwise be a window that
never opens. Once the GUI is up it reports its own errors.
"""

import sys
import traceback


def report(message: str) -> None:
    """Say something went wrong, given there is nowhere to print it."""
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("nammy", message)
        root.destroy()
    except Exception:
        # Tk itself is missing or broken, so the package is not merely misbuilt.
        # There is no console to fall back to either; the exit code is the report.
        pass


def main() -> None:
    try:
        from nammy.gui import main as gui
    except Exception:
        report(f"nammy could not start.\n\n{traceback.format_exc()}")
        raise SystemExit(1)
    gui()


if __name__ == "__main__":
    sys.exit(main())
