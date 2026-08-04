"""
nammy: a tinygrad-based trainer for NAM's classic ("A1") WaveNet architecture.

Usage:
    uv run python -m nammy train input.wav output.wav [--epochs N] [--out model.nam] ...
    uv run python -m nammy process model.nam input.wav output.wav
    uv run python -m nammy gui
"""

import argparse

from nammy.data import load_pair, read_wav, write_wav
from nammy.train import train
from nammy.wavenet import WaveNet


def cmd_train(args):
    x, y, sample_rate = load_pair(args.input, args.output, latency=args.latency)
    print(f"loaded {len(x)} samples @ {sample_rate} Hz")

    model = WaveNet()
    # train() writes args.out whenever the best checkpoint improves, so an
    # interrupted run still leaves the best model so far on disk.
    try:
        train(
            model,
            x,
            y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            ny=args.ny,
            lr=args.lr,
            out=args.out,
            sample_rate=float(sample_rate),
        )
    except KeyboardInterrupt:
        print(f"\ninterrupted; best checkpoint so far is in {args.out}")
        return
    print(f"exported {args.out}")


def cmd_process(args):
    model, model_rate = WaveNet.from_nam(args.model)
    x, rate = read_wav(args.input)
    if model_rate is not None and rate != model_rate:
        print(f"warning: model was trained at {model_rate} Hz, input is {rate} Hz")
    print(f"processing {len(x)} samples @ {rate} Hz")
    y = model.process(x)
    write_wav(args.output, y, rate)
    print(f"wrote {args.output}")


def cmd_gui(args):
    # Imported here so the CLI does not pay for tkinter, which may be absent.
    from nammy.gui import main as gui_main

    gui_main()


def main():
    parser = argparse.ArgumentParser(description="NAM WaveNet (A1) trainer on tinygrad")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model from an input/output WAV pair")
    p_train.add_argument("input", help="Input (DI/reamp source) WAV")
    p_train.add_argument("output", help="Output (amp-processed) WAV")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--ny", type=int, default=8192)
    p_train.add_argument("--lr", type=float, default=0.004)
    p_train.add_argument("--latency", type=int, default=0, help="Output latency in samples")
    p_train.add_argument("--out", default="model.nam", help="Path for the exported .nam")
    p_train.set_defaults(func=cmd_train)

    p_proc = sub.add_parser("process", help="Run a WAV through a .nam model (reamp)")
    p_proc.add_argument("model", help="Path to a WaveNet .nam file")
    p_proc.add_argument("input", help="Input WAV to process")
    p_proc.add_argument("output", help="Where to write the processed WAV (24-bit PCM)")
    p_proc.set_defaults(func=cmd_process)

    p_gui = sub.add_parser("gui", help="Open the Tkinter GUI")
    p_gui.set_defaults(func=cmd_gui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
