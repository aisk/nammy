"""
WAV loading and dataset slicing, mirroring nam.data.Dataset semantics.
"""

from __future__ import annotations

import random
import struct
import wave

try:
    import numpy as np
except ImportError:  # a build with no compiled dependencies
    from . import _numpy_compat as np


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """
    Read a mono WAV file (PCM 16/24/32-bit or IEEE float32/64) as float32 in [-1, 1].

    :return: (samples, sample_rate)
    """
    try:
        with wave.open(path, "rb") as fp:
            params = fp.getparams()
            raw = fp.readframes(params.nframes)
        return _decode_pcm(raw, params.sampwidth, params.nchannels), params.framerate
    except wave.Error:
        # stdlib wave rejects IEEE-float WAVs; fall back to a minimal RIFF parser.
        return _read_wav_riff(path)


def _decode_pcm(raw: bytes, sampwidth: int, nchannels: int) -> np.ndarray:
    if sampwidth == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 2**15
    elif sampwidth == 3:
        # There is no 24-bit dtype, so rebuild the samples from three byte
        # planes. Stepped slicing of bytes copies at C speed, and the sign of
        # a little-endian sample lives entirely in its top byte.
        raw = raw[: len(raw) // 3 * 3]
        low, mid, high = raw[0::3], raw[1::3], raw[2::3]
        x = (
            np.frombuffer(low, dtype=np.uint8).astype(np.int32)
            | (np.frombuffer(mid, dtype=np.uint8).astype(np.int32) << 8)
            | (np.frombuffer(high, dtype=np.int8).astype(np.int32) << 16)
        )
        x = x.astype(np.float32) / 2**23
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2**31
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    if nchannels > 1:
        x = x[::nchannels].copy()
    return x


def _read_wav_riff(path: str) -> tuple[np.ndarray, int]:
    with open(path, "rb") as fp:
        data = fp.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")
    pos, fmt, payload = 12, None, None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body = data[pos + 8 : pos + 8 + size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError(f"Missing fmt/data chunk: {path}")
    audio_format, nchannels, framerate, _, _, bits = fmt
    if audio_format == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: treat by bit depth
        audio_format = 3 if bits in (32, 64) else 1
    if audio_format == 3:  # IEEE float
        dtype = "<f4" if bits == 32 else "<f8"
        x = np.frombuffer(payload[: len(payload) // (bits // 8) * (bits // 8)], dtype=dtype)
        x = x.astype(np.float32)
        if nchannels > 1:
            x = x[::nchannels].copy()
        return x, framerate
    elif audio_format == 1:
        return _decode_pcm(payload, bits // 8, nchannels), framerate
    raise ValueError(f"Unsupported WAV format {audio_format}")


def write_wav(path: str, x: np.ndarray, rate: int, sampwidth: int = 3) -> None:
    """Write mono float samples in [-1, 1] as PCM (16- or 24-bit, like NAM's default)."""
    x = np.clip(x, -1.0, 1.0)
    if sampwidth == 2:
        frames = (x * (2**15 - 1)).astype("<i2").tobytes()
    elif sampwidth == 3:
        # A 24-bit sample is a little-endian int32 without its top byte.
        buf = bytearray((x * (2**23 - 1)).astype("<i4").tobytes())
        del buf[3::4]
        frames = bytes(buf)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    with wave.open(path, "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(sampwidth)
        fp.setframerate(rate)
        fp.writeframes(frames)


class Dataset:
    """
    Slices aligned (x, y) into training pairs like nam.data.Dataset:
    item i is (x[i*ny : i*ny + nx+ny-1], y[i*ny + nx-1 : i*ny + nx-1 + ny]).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, nx: int, ny: int):
        n = min(len(x), len(y))
        self.x = x[:n].astype(np.float32)
        self.y = y[:n].astype(np.float32)
        self.nx = nx
        self.ny = ny
        if len(self) < 1:
            raise ValueError(f"Not enough data: {n} samples for nx={nx}, ny={ny}")

    def __len__(self) -> int:
        return (len(self.x) - self.nx + 1) // self.ny

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        i = idx * self.ny
        j = i + self.nx - 1
        return self.x[i : i + self.nx + self.ny - 1], self.y[j : j + self.ny]

    def n_batches(self, batch_size: int) -> int:
        """How many batches an epoch yields, given that the ragged tail is dropped."""
        return max((len(self) - batch_size) // batch_size + 1, 0)

    def batches(self, batch_size: int, rng: random.Random):
        """Yield (X, Y) numpy batches over a shuffled epoch; drops the ragged tail."""
        order = list(range(len(self)))
        rng.shuffle(order)
        for start in range(0, self.n_batches(batch_size) * batch_size, batch_size):
            idxs = order[start : start + batch_size]
            xs, ys = zip(*(self[i] for i in idxs))
            yield np.stack(xs).reshape(len(xs), 1, -1), np.stack(ys)


def load_pair(
    x_path: str, y_path: str, latency: int = 0
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load input/output WAVs, compensate latency, return (x, y, sample_rate)."""
    x, rate_x = read_wav(x_path)
    y, rate_y = read_wav(y_path)
    if rate_x != rate_y:
        raise ValueError(f"Sample rate mismatch: {rate_x} vs {rate_y}")
    if latency > 0:
        y = y[latency:]
    elif latency < 0:
        x = x[-latency:]
    n = min(len(x), len(y))
    return x[:n], y[:n], rate_x
