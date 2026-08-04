"""
Tkinter front end for nammy: train a model from a WAV pair, or reamp through one.

Start it with `python -m nammy gui`, or by double-clicking nammy-gui.pyw.

Tk's main loop must stay responsive, so the work runs on a daemon thread that
only ever posts messages onto a queue; the main thread drains that queue on a
timer and is the sole caller of every Tk method.
"""

from __future__ import annotations

import json
import math
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from .data import load_pair, read_wav, write_wav
from .train import train
from .wavenet import WaveNet

POLL_MS = 100
LOG_MAX_LINES = 4000
SMALL_FONT = ("TkDefaultFont", 7)


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def _fixed_font():
    """TkFixedFont, moved off Courier New where something tighter is installed."""
    font = tkfont.nametofont("TkFixedFont").copy()
    available = set(tkfont.families())
    for family in ("Consolas", "Menlo", "DejaVu Sans Mono", "Liberation Mono"):
        if family in available:
            font.configure(family=family)
            break
    return font


class EsrPlot(tk.Canvas):
    """Validation ESR against epoch, on a log axis, drawn with plain canvas lines."""

    PAD_L, PAD_R, PAD_T, PAD_B = 46, 8, 8, 14

    def __init__(self, master, **kw):
        super().__init__(
            master,
            height=110,
            background="white",
            highlightthickness=1,
            highlightbackground="#c8c8c8",
            **kw,
        )
        self._values: list[float] = []
        self._total = 0
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_data(self, values, total: int) -> None:
        self._values = list(values)
        self._total = max(total, len(self._values))
        self._redraw()

    def clear(self) -> None:
        self.set_data([], 0)

    def _redraw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 1 or h <= 1:
            return
        x0, x1 = self.PAD_L, w - self.PAD_R
        y0, y1 = self.PAD_T, h - self.PAD_B
        if x1 <= x0 or y1 <= y0:
            return
        good = [v for v in self._values if v > 0 and math.isfinite(v)]
        if not good:
            self.create_text(
                (x0 + x1) // 2,
                (y0 + y1) // 2,
                text="no validation data yet",
                fill="#909090",
                font=SMALL_FONT,
            )
            return

        lo, hi = min(good), max(good)
        if hi <= lo:
            lo, hi = lo / 3.0, hi * 3.0
        llo, lhi = math.log10(lo), math.log10(hi)
        span = lhi - llo

        def sy(v: float) -> float:
            return y1 - (math.log10(v) - llo) / span * (y1 - y0)

        n = max(self._total - 1, 1)

        def sx(i: int) -> float:
            return x0 + i / n * (x1 - x0)

        for e in range(math.floor(llo), math.ceil(lhi) + 1):
            decade = 10.0**e
            if lo <= decade <= hi:
                self.create_line(x0, sy(decade), x1, sy(decade), fill="#ededed")
        self.create_rectangle(x0, y0, x1, y1, outline="#d8d8d8")

        best = min(good)
        self.create_line(x0, sy(best), x1, sy(best), fill="#4a90d9", dash=(3, 3))
        for v, anchor in ((hi, "e"), (lo, "e")):
            self.create_text(x0 - 5, sy(v), text=_fmt(v), anchor=anchor, font=SMALL_FONT)

        points: list[float] = []
        for i, v in enumerate(self._values):
            if v > 0 and math.isfinite(v):
                points.extend([sx(i), sy(v)])
        if len(points) >= 4:
            self.create_line(*points, fill="#c0392b", width=2)
        elif len(points) == 2:
            px, py = points
            self.create_oval(px - 2, py - 2, px + 2, py + 2, fill="#c0392b", outline="")

        self.create_text(
            x1,
            y1 + 1,
            text=f"epoch {len(self._values)}/{self._total} · best {_fmt(best)}",
            anchor="ne",
            fill="#606060",
            font=SMALL_FONT,
        )


class _JobTab(ttk.Frame):
    """A tab that runs at most one background job and streams its output to the UI."""

    def __init__(self, master, set_status):
        super().__init__(master, padding=10)
        self._set_status = set_status
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.after(POLL_MS, self._drain)

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request_stop(self) -> None:
        self._stop.set()

    def post(self, kind: str, payload=None) -> None:
        """Hand a message to the UI thread. Safe to call from the worker."""
        self._queue.put((kind, payload))

    def start_job(self, fn) -> None:
        self._stop.clear()

        def run():
            try:
                self.post("done", (fn(), None))
            except Exception as exc:  # reported in the log, not to a dead console
                self.post("done", (None, exc))

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        try:
            while True:
                try:
                    kind, payload = self._queue.get_nowait()
                except queue.Empty:
                    break
                self.handle(kind, payload)
        finally:
            # Always reschedule: a handler that raises must not freeze the stream.
            self.after(POLL_MS, self._drain)

    def handle(self, kind: str, payload) -> None:
        raise NotImplementedError

    # -- shared widgets ----------------------------------------------------

    def _file_row(self, row: int, label: str, var: tk.StringVar, command) -> None:
        ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(self, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        button = ttk.Button(self, text="Browse…", command=command, width=10)
        button.grid(row=row, column=2, pady=2)

    def _make_log(self, row: int) -> None:
        frame = ttk.LabelFrame(self, text="Log", padding=4)
        frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.log = tk.Text(frame, height=8, wrap="word", font=_fixed_font(), state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=bar.set)
        self.rowconfigure(row, weight=1)

    def append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip("\n") + "\n")
        excess = int(self.log.index("end-1c").split(".")[0]) - LOG_MAX_LINES
        if excess > 0:
            self.log.delete("1.0", f"{excess + 1}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


class TrainTab(_JobTab):
    def __init__(self, master, set_status):
        super().__init__(master, set_status)
        self.columnconfigure(1, weight=1)

        self.v_input = tk.StringVar()
        self.v_target = tk.StringVar()
        self.v_out = tk.StringVar(value="model.nam")
        self._file_row(0, "Input WAV (DI)", self.v_input, self._pick_input)
        self._file_row(1, "Target WAV (amp)", self.v_target, self._pick_target)
        self._file_row(2, "Output .nam", self.v_out, self._pick_out)

        self.v_epochs = tk.StringVar(value="100")
        self.v_batch = tk.StringVar(value="16")
        self.v_ny = tk.StringVar(value="8192")
        self.v_lr = tk.StringVar(value="0.004")
        self.v_latency = tk.StringVar(value="0")
        params = ttk.LabelFrame(self, text="Parameters", padding=6)
        params.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self._param(params, 0, 0, "Epochs", self.v_epochs)
        self._param(params, 0, 2, "Batch size", self.v_batch)
        self._param(params, 0, 4, "ny", self.v_ny)
        self._param(params, 1, 0, "Learning rate", self.v_lr)
        self._param(params, 1, 2, "Latency (samples)", self.v_latency)
        ttk.Label(params, text="Device").grid(row=1, column=4, sticky="e", padx=(12, 4), pady=2)
        self.device_label = ttk.Label(params, text="…", foreground="#606060")
        self.device_label.grid(row=1, column=5, sticky="w", pady=2)
        params.columnconfigure(6, weight=1)

        controls = ttk.Frame(self)
        controls.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        controls.columnconfigure(2, weight=1)
        self.start_button = ttk.Button(controls, text="Start", command=self._start, width=10)
        self.start_button.grid(row=0, column=0)
        self.stop_button = ttk.Button(
            controls, text="Stop", command=self._request_stop, width=10, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=6)
        self.summary = ttk.Label(controls, text="", anchor="e")
        self.summary.grid(row=0, column=2, sticky="e")

        self.progress = ttk.Progressbar(self, maximum=1.0)
        self.progress.grid(row=5, column=0, columnspan=3, sticky="ew")

        plot_frame = ttk.LabelFrame(self, text="Validation ESR", padding=4)
        plot_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        plot_frame.columnconfigure(0, weight=1)
        self.plot = EsrPlot(plot_frame)
        self.plot.grid(row=0, column=0, sticky="ew")

        self._make_log(7)

        self._esr_history: list[float] = []
        self._epochs_done = 0
        self._total_epochs = 0
        self._inputs = [
            w
            for w in list(self.grid_slaves()) + list(params.grid_slaves())
            if isinstance(w, (ttk.Entry, ttk.Button))
        ]
        threading.Thread(target=self._probe_device, daemon=True).start()

    @staticmethod
    def _param(parent, row: int, col: int, label: str, var: tk.StringVar) -> None:
        pad = (12, 4) if col else (0, 4)
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="e", padx=pad, pady=2)
        ttk.Entry(parent, textvariable=var, width=9).grid(row=row, column=col + 1, sticky="w")

    def _probe_device(self) -> None:
        """Naming the device initializes it, so keep it off the startup path."""
        try:
            from tinygrad import Device

            self.post("device", Device.DEFAULT)
        except Exception:
            self.post("device", "?")

    # -- file pickers ------------------------------------------------------

    def _pick_input(self):
        self._ask_wav(self.v_input, "Select the input (DI) WAV")

    def _pick_target(self):
        self._ask_wav(self.v_target, "Select the target (amp-processed) WAV")

    def _ask_wav(self, var: tk.StringVar, title: str) -> None:
        path = filedialog.askopenfilename(
            parent=self, title=title, filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            var.set(path)

    def _pick_out(self):
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Where to write the model",
            defaultextension=".nam",
            initialfile=self.v_out.get() or "model.nam",
            filetypes=[("NAM models", "*.nam"), ("All files", "*.*")],
        )
        if path:
            self.v_out.set(path)

    # -- running -----------------------------------------------------------

    def _read_params(self) -> dict:
        if not self.v_input.get() or not self.v_target.get():
            raise ValueError("Pick both an input and a target WAV first.")
        if not self.v_out.get():
            raise ValueError("Pick where to write the .nam file.")
        return {
            "input": self.v_input.get(),
            "target": self.v_target.get(),
            "out": self.v_out.get(),
            "epochs": _num(self.v_epochs.get(), "Epochs", int, minimum=1),
            "batch_size": _num(self.v_batch.get(), "Batch size", int, minimum=1),
            "ny": _num(self.v_ny.get(), "ny", int, minimum=1),
            "lr": _num(self.v_lr.get(), "Learning rate", float, minimum=0.0),
            "latency": _num(self.v_latency.get(), "Latency", int),
        }

    def _start(self) -> None:
        if self.busy:
            return
        try:
            params = self._read_params()
        except ValueError as exc:
            messagebox.showerror("nammy", str(exc), parent=self)
            return
        self.clear_log()
        self.plot.clear()
        self._esr_history = []
        self._epochs_done = 0
        self._total_epochs = params["epochs"]
        self.progress.configure(value=0.0)
        self.summary.configure(text="")
        self._set_running(True)
        self._set_status("Training…")
        self.start_job(lambda: self._run(params))

    def _request_stop(self) -> None:
        self.stop_button.configure(state="disabled")
        self._set_status("Stopping after the current batch…")
        self.request_stop()

    def _run(self, p: dict) -> dict:
        x, y, rate = load_pair(p["input"], p["target"], latency=p["latency"])
        self.post("log", f"loaded {len(x)} samples @ {rate} Hz")
        model = WaveNet()
        # train() rewrites p["out"] on every improvement, so stopping early still
        # leaves the best model so far on disk.
        return train(
            model,
            x,
            y,
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            ny=p["ny"],
            lr=p["lr"],
            out=p["out"],
            sample_rate=float(rate),
            log=lambda m: self.post("log", m),
            should_stop=self._stop.is_set,
            on_epoch=lambda info: self.post("epoch", info),
            on_batch=lambda done, total: self.post("batch", (done, total)),
        )

    def _set_running(self, running: bool) -> None:
        for widget in self._inputs:
            widget.configure(state="disabled" if running else "normal")
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def handle(self, kind: str, payload) -> None:
        if kind == "log":
            self.append_log(payload)
        elif kind == "device":
            self.device_label.configure(text=payload)
        elif kind == "batch":
            done, total = payload
            if total:
                self.progress.configure(
                    value=(self._epochs_done + done / total) / max(self._total_epochs, 1)
                )
        elif kind == "epoch":
            self._epochs_done = payload["epoch"]
            self._esr_history.append(payload["val_esr"])
            self.plot.set_data(self._esr_history, payload["epochs"])
            self.progress.configure(value=payload["epoch"] / max(payload["epochs"], 1))
            self.summary.configure(
                text=f"epoch {payload['epoch']}/{payload['epochs']} · "
                f"best ESR {payload['best_esr']:.4f} · {payload['seconds']:.1f} s/epoch"
            )
        elif kind == "done":
            result, error = payload
            self._set_running(False)
            if error is not None:
                self.append_log(f"error: {error}")
                self._set_status(f"Failed: {error}")
                messagebox.showerror("nammy", str(error), parent=self)
            elif result.get("stopped"):
                self._set_status(f"Stopped. Best model so far is in {self.v_out.get()}")
            else:
                self._set_status(
                    f"Done. Best val ESR {result['best_esr']:.4f}, written to {self.v_out.get()}"
                )


class ProcessTab(_JobTab):
    def __init__(self, master, set_status):
        super().__init__(master, set_status)
        self.columnconfigure(1, weight=1)

        self.v_model = tk.StringVar()
        self.v_input = tk.StringVar()
        self.v_output = tk.StringVar()
        self._file_row(0, "Model (.nam)", self.v_model, self._pick_model)
        self._file_row(1, "Input WAV", self.v_input, self._pick_input)
        self._file_row(2, "Output WAV", self.v_output, self._pick_output)

        controls = ttk.Frame(self)
        controls.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        controls.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(controls, text="Process", command=self._start, width=10)
        self.start_button.grid(row=0, column=0)
        self.model_label = ttk.Label(controls, text="", anchor="e", foreground="#606060")
        self.model_label.grid(row=0, column=1, sticky="e")

        self.progress = ttk.Progressbar(self, maximum=1.0)
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew")

        self._make_log(5)
        self._inputs = [
            w for w in self.grid_slaves() if isinstance(w, (ttk.Entry, ttk.Button))
        ]

    def _pick_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Select a .nam model",
            filetypes=[("NAM models", "*.nam"), ("All files", "*.*")],
        )
        if path:
            self.v_model.set(path)
            self.model_label.configure(text=_describe_nam(path))

    def _pick_input(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Select the WAV to process",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.v_input.set(path)

    def _pick_output(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Where to write the processed WAV",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.v_output.set(path)

    def _start(self) -> None:
        if self.busy:
            return
        if not (self.v_model.get() and self.v_input.get() and self.v_output.get()):
            messagebox.showerror("nammy", "Pick a model, an input and an output.", parent=self)
            return
        self.clear_log()
        self.progress.configure(value=0.0)
        self._set_running(True)
        self._set_status("Processing…")
        paths = (self.v_model.get(), self.v_input.get(), self.v_output.get())
        self.start_job(lambda: self._run(*paths))

    def _run(self, model_path: str, in_path: str, out_path: str) -> str:
        model, model_rate = WaveNet.from_nam(model_path)
        x, rate = read_wav(in_path)
        if model_rate is not None and rate != model_rate:
            self.post("log", f"warning: model was trained at {model_rate} Hz, input is {rate} Hz")
        self.post("log", f"processing {len(x)} samples @ {rate} Hz")
        y = model.process(x, progress=lambda done, total: self.post("progress", done / total))
        write_wav(out_path, y, rate)
        self.post("log", f"wrote {out_path}")
        return out_path

    def _set_running(self, running: bool) -> None:
        for widget in self._inputs:
            widget.configure(state="disabled" if running else "normal")
        self.start_button.configure(state="disabled" if running else "normal")

    def handle(self, kind: str, payload) -> None:
        if kind == "log":
            self.append_log(payload)
        elif kind == "progress":
            self.progress.configure(value=payload)
        elif kind == "done":
            result, error = payload
            self._set_running(False)
            if error is not None:
                self.append_log(f"error: {error}")
                self._set_status(f"Failed: {error}")
                messagebox.showerror("nammy", str(error), parent=self)
            else:
                self.progress.configure(value=1.0)
                self._set_status(f"Done. Wrote {result}")


def _num(text: str, name: str, cast, minimum=None):
    try:
        value = cast(text.strip())
    except (ValueError, AttributeError):
        raise ValueError(f"{name}: {text!r} is not a number.") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _describe_nam(path: str) -> str:
    """One-line summary of a .nam, read straight from the json (no device work)."""
    try:
        with open(path) as fp:
            d = json.load(fp)
    except Exception as exc:
        return f"unreadable: {exc}"
    rate = d.get("sample_rate")
    rate = f"{rate:g} Hz" if isinstance(rate, (int, float)) else "unknown rate"
    return f"{d.get('architecture', '?')} · {rate}"


def _enable_dpi_awareness() -> None:
    """Ask Windows for real pixels, so Tk renders sharp instead of bitmap-stretched."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main() -> None:
    _enable_dpi_awareness()
    root = tk.Tk()
    root.title("nammy")
    root.minsize(660, 600)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    status = tk.StringVar(value="Ready.")

    notebook = ttk.Notebook(root)
    notebook.grid(row=0, column=0, sticky="nsew")
    tabs = [TrainTab(notebook, status.set), ProcessTab(notebook, status.set)]
    notebook.add(tabs[0], text="  Train  ")
    notebook.add(tabs[1], text="  Process  ")

    ttk.Label(root, textvariable=status, relief="sunken", anchor="w", padding=(6, 3)).grid(
        row=1, column=0, sticky="ew"
    )

    def on_close():
        if any(tab.busy for tab in tabs):
            if not messagebox.askokcancel("nammy", "A job is still running. Quit anyway?"):
                return
            for tab in tabs:
                tab.request_stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
