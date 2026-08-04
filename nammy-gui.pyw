"""
Double-clickable launcher for the nammy GUI.

Windows runs .pyw without a console, which is what you want for a GUI but also
means a traceback would go nowhere, so failures are reported in a message box.
Explorer starts this with whichever interpreter is registered for .pyw, and that
is usually not the project's virtualenv, so re-exec into .venv first when there
is one. Deliberately plain: this may run under an older system Python than the
package itself requires.
"""

import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
GUARD = "NAMMY_GUI_RELAUNCHED"
VENV_CANDIDATES = (
    os.path.join("Scripts", "pythonw.exe"),
    os.path.join("Scripts", "python.exe"),
    os.path.join("bin", "python"),
)


def venv_interpreter():
    for rel in VENV_CANDIDATES:
        exe = os.path.join(ROOT, ".venv", rel)
        if os.path.exists(exe):
            return exe
    return None


def relaunch_in_venv():
    """Replace this process with the project virtualenv's interpreter, if any."""
    exe = venv_interpreter()
    if exe is None or os.environ.get(GUARD):
        return
    running = os.path.dirname(os.path.realpath(sys.executable))
    if running == os.path.dirname(os.path.realpath(exe)):
        return
    os.environ[GUARD] = "1"
    os.execv(exe, [exe, os.path.abspath(__file__)])


def report(message):
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("nammy", message)
        root.destroy()
    except Exception:
        sys.stderr.write(message + "\n")


def main():
    relaunch_in_venv()
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from nammy.gui import main as gui_main
    except Exception:
        report(
            "Could not start the GUI with {0}.\n\n{1}".format(
                sys.executable, traceback.format_exc()
            )
        )
        raise SystemExit(1)
    gui_main()


if __name__ == "__main__":
    main()
