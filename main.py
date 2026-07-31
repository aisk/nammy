"""
nammy: a tinygrad-based trainer for NAM's classic ("A1") WaveNet architecture.

Usage:
    uv run main.py input.wav output.wav [--epochs N] [--ny N] [--batch-size N]
                   [--latency SAMPLES] [--out model.nam]
"""

import argparse

from nammy.data import load_pair
from nammy.train import train
from nammy.wavenet import WaveNet


def main():
    parser = argparse.ArgumentParser(description="Train a NAM WaveNet (A1) with tinygrad")
    parser.add_argument("input", help="Input (DI/reamp source) WAV")
    parser.add_argument("output", help="Output (amp-processed) WAV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ny", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--latency", type=int, default=0, help="Output latency in samples")
    parser.add_argument("--out", default="model.nam", help="Path for the exported .nam")
    args = parser.parse_args()

    x, y, sample_rate = load_pair(args.input, args.output, latency=args.latency)
    print(f"loaded {len(x)} samples @ {sample_rate} Hz")

    model = WaveNet()
    train(
        model,
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ny=args.ny,
        lr=args.lr,
    )
    model.export_nam(args.out, sample_rate=float(sample_rate))
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
