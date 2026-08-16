"""
Draw nammy's icons: the MSIX tile images and the .ico PyInstaller stamps on the exe.

    python tools/make_assets.py            # -> packaging/assets/*.png, packaging/nammy.ico

The mark is a rounded tile holding a row of bars whose heights are a signal
driven through a tanh: they climb steadily on the way in and flatten off at the
top, which is the one thing an amp does that a graph can show. Bars rather than
a drawn waveform because a curve sampled per column draws its steep parts as a
hairline, and the smallest tile here is 24 px.

It is generated rather than checked in as binaries because it is a dozen
rectangles, and every size has to say the same thing. Everything is drawn at 4x
into a one-byte-per-sample coverage buffer and box filtered down, which is all
the antialiasing shapes this simple need and keeps the build free of any
imaging dependency. Replace this file wholesale if you have a real icon;
nothing else cares where the PNGs came from.
"""

from __future__ import annotations

import math
import pathlib
import struct
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "packaging" / "assets"
ICON = ROOT / "packaging" / "nammy.ico"

SUPERSAMPLE = 4
BACKGROUND = (0x17, 0x1C, 0x26)  # also the manifest's tile BackgroundColor
WAVE = (0xFF, 0xB3, 0x40)

BARS = 5
GAP = 0.55  # spacing between bars, as a fraction of a bar's width
DRIVE = 2.8  # how hard the last bar is pushed into the tanh
SHORTEST = 0.34  # height of the first bar, as a fraction of the tallest

# Tiles the manifest names, all unqualified: scale- and targetsize-qualified
# filenames would need a resources.pri beside them, and Windows is content to
# rescale these.
TILES = {
    "StoreLogo.png": (50, 50),
    "Square44x44Logo.png": (44, 44),
    "Square71x71Logo.png": (71, 71),
    "Square150x150Logo.png": (150, 150),
    "Square310x310Logo.png": (310, 310),
    "Wide310x150Logo.png": (310, 150),
}
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)

EMPTY, TILE, LINE = 0, 1, 2


def levels() -> list[float]:
    """
    Bar heights from 0 to 1, a ramp through a tanh.

    Normalised so the tallest bar is exactly 1 and the shortest is SHORTEST,
    which fixes the shape of the curve independently of DRIVE: raising the drive
    then only changes how early the bars stop growing, which is the point.
    """
    ramp = [math.tanh(DRIVE * (i + 1) / BARS) / math.tanh(DRIVE) for i in range(BARS)]
    return [SHORTEST + (1 - SHORTEST) * (v - ramp[0]) / (ramp[-1] - ramp[0]) for v in ramp]


def render(width: int, height: int) -> bytearray:
    """A width*height RGBA buffer, box filtered down from 4x."""
    ss = SUPERSAMPLE
    w, h = width * ss, height * ss
    buffer = bytearray(w * h)

    side = min(w, h)
    _fill(buffer, w, h, 0, 0, w, h, 0.16 * side, TILE)

    margin = 0.18 * side
    bar = (w - 2 * margin) / (BARS + (BARS - 1) * GAP)  # bars and gaps share the width
    tallest = h - 2 * margin
    for index, level in enumerate(levels()):
        length = max(level * tallest, bar)  # a bar is never shorter than it is wide
        x = margin + index * bar * (1 + GAP)
        _fill(buffer, w, h, x, h / 2 - length / 2, bar, length, bar / 2, LINE)

    return _downsample(buffer, w, h, ss)


def _fill(
    buffer: bytearray,
    w: int,
    h: int,
    x: float,
    y: float,
    width: float,
    height: float,
    radius: float,
    value: int,
) -> None:
    """Paint a rounded rectangle, clipped to the buffer, over whatever is there."""
    radius = min(radius, width / 2, height / 2)
    for py in range(max(0, int(y)), min(h, int(math.ceil(y + height)))):
        # Distance from the row to the nearer of the two rows of corner centres,
        # which is what decides how far the rectangle reaches across this row.
        dy = max(y + radius - (py + 0.5), (py + 0.5) - (y + height - radius), 0.0)
        if dy > radius:
            continue
        inset = radius - math.sqrt(radius * radius - dy * dy) if radius else 0.0
        row = py * w
        for px in range(max(0, int(x + inset)), min(w, int(math.ceil(x + width - inset)))):
            buffer[row + px] = value


def _downsample(buffer: bytearray, w: int, h: int, ss: int) -> bytearray:
    """Average each ss*ss block into one RGBA pixel, unpremultiplied."""
    out = bytearray(w // ss * h // ss * 4)
    cell = ss * ss
    at = 0
    for y in range(0, h, ss):
        for x in range(0, w, ss):
            r = g = b = alpha = 0
            for dy in range(ss):
                row = (y + dy) * w + x
                for dx in range(ss):
                    sample = buffer[row + dx]
                    if sample == EMPTY:
                        continue
                    alpha += 1
                    colour = WAVE if sample == LINE else BACKGROUND
                    r += colour[0]
                    g += colour[1]
                    b += colour[2]
            if alpha:
                out[at : at + 4] = bytes((r // alpha, g // alpha, b // alpha, alpha * 255 // cell))
            at += 4
    return out


def png(pixels: bytearray, width: int, height: int) -> bytes:
    """An RGBA PNG. Written by hand so the build needs nothing installed."""
    stride = width * 4
    raw = b"".join(b"\x00" + bytes(pixels[y * stride : (y + 1) * stride]) for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + b"".join(
        _chunk(kind, data)
        for kind, data in (
            (b"IHDR", header),
            (b"IDAT", zlib.compress(raw, 9)),
            (b"IEND", b""),
        )
    )


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def ico(images: list[tuple[int, bytearray]]) -> bytes:
    """
    A multi-size .ico of 32-bit DIB entries.

    PNG-compressed entries would be a third of the size and every Windows since
    Vista reads them, but PyInstaller copies icon entries into the exe's
    resources verbatim and plain DIBs are what every consumer of those agrees
    on, so this writes the boring format.
    """
    directory, blobs, offset = [], [], 6 + 16 * len(images)
    for size, pixels in images:
        blob = _dib(pixels, size)
        directory.append(
            struct.pack("<BBBBHHII", size & 0xFF, size & 0xFF, 0, 0, 1, 32, len(blob), offset)
        )
        blobs.append(blob)
        offset += len(blob)
    return struct.pack("<HHH", 0, 1, len(images)) + b"".join(directory) + b"".join(blobs)


def _dib(pixels: bytearray, size: int) -> bytes:
    """BITMAPINFOHEADER, BGRA rows bottom-up, then the AND mask the format still wants."""
    head = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0, size * size * 4, 0, 0, 0, 0)
    rows = []
    for y in range(size - 1, -1, -1):
        row = bytearray(size * 4)
        for x in range(size):
            r, g, b, a = pixels[(y * size + x) * 4 : (y * size + x) * 4 + 4]
            row[x * 4 : x * 4 + 4] = bytes((b, g, r, a))
        rows.append(bytes(row))
    mask = bytes((size + 31) // 32 * 4 * size)  # fully opaque; the alpha channel rules
    return head + b"".join(rows) + mask


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, (width, height) in TILES.items():
        path = ASSETS / name
        path.write_bytes(png(render(width, height), width, height))
        print(f"{path.relative_to(ROOT)} ({width}x{height})")

    ICON.write_bytes(ico([(size, render(size, size)) for size in ICON_SIZES]))
    print(f"{ICON.relative_to(ROOT)} ({', '.join(str(s) for s in ICON_SIZES)})")


if __name__ == "__main__":
    main()
