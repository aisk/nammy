"""
A stand-in for the slice of numpy that nammy uses, so the project can run with
no compiled dependencies at all.

This is not a numpy clone. It implements exactly what nammy and tinygrad reach
for and raises rather than guessing anywhere else. Two choices keep it small:

* Storage is one flat `array.array` plus a shape tuple, and every index other
  than a whole-array reshape copies. nammy never writes through a view, so the
  strided-view machinery that is most of real numpy is not needed here.
* Bulk arithmetic is handed to tinygrad, which is a hard dependency already and
  is already compiled. Only byte-level work (slicing, stacking, filling,
  reinterpreting) happens in this file, where `array` does it at C speed.

Byte order is native throughout. The "<i2"-style codes nammy passes are read
for their width and signedness only, which is correct on the little-endian
platforms it runs on.
"""

from __future__ import annotations

import array as _array
import math
import operator
import sys


class _DType:
    """Just enough of numpy.dtype for tinygrad's dtype lookups."""

    __slots__ = ("name", "code")

    def __init__(self, name: str, code: str):
        self.name = name  # tinygrad indexes its own dtype table by this
        self.code = code  # array.array typecode

    @property
    def itemsize(self) -> int:
        return _array.array(self.code).itemsize

    @property
    def type(self) -> "_DType":
        # numpy hands back a scalar type here, but tinygrad only feeds it
        # straight back into frombuffer, so the dtype itself will do.
        return self

    def __repr__(self) -> str:
        return f"dtype('{self.name}')"


int8 = _DType("int8", "b")
uint8 = _DType("uint8", "B")
int16 = _DType("int16", "h")
int32 = _DType("int32", "i")
float32 = _DType("float32", "f")
float64 = _DType("float64", "d")

_BY_CODE = {
    "<i2": int16,
    "<i4": int32,
    "<f4": float32,
    "<f8": float64,
    "b": int8,
    "B": uint8,
    "h": int16,
    "i": int32,
    "f": float32,
    "d": float64,
}

# array typecodes are C types, so their widths are worth confirming rather than
# assuming: everything below reinterprets raw bytes through them.
assert (int8.itemsize, int16.itemsize, int32.itemsize) == (1, 2, 4)
assert (float32.itemsize, float64.itemsize) == (4, 8)


def _dt(spec) -> _DType:
    """Resolve a dtype spec. Exposed as numpy.dtype, which tinygrad calls."""
    if isinstance(spec, _DType):
        return spec
    try:
        return _BY_CODE[spec]
    except (KeyError, TypeError):
        raise NotImplementedError(f"unsupported dtype {spec!r}") from None


dtype = _dt


def _to_tensor(a: "ndarray"):
    """Upload for the bulk arithmetic this file deliberately does not do."""
    from tinygrad import Tensor, dtypes

    from . import device

    # Arithmetic needs a backend, and reading a WAV is allowed to be the thing
    # that settles it; ensure_selected() is a no-op once a choice was made.
    device.ensure_selected()
    return Tensor(a.tobytes(), dtype=getattr(dtypes, a.dtype.name))


def _from_tensor(t, shape: tuple[int, ...] | None = None) -> "ndarray":
    # tinygrad names its dtypes after C types ("float", "signed char"), but its
    # struct code is exactly the array typecode wanted here.
    dt = _BY_CODE[t.dtype.fmt]
    buf = _array.array(dt.code)
    buf.frombytes(t.data().cast("B"))
    return ndarray(buf, dt, shape if shape is not None else (len(buf),))


def _resolve(shape: tuple[int, ...], size: int) -> tuple[int, ...]:
    """Fill in the one -1 a reshape is allowed to leave open."""
    if -1 not in shape:
        return shape
    known = math.prod(d for d in shape if d != -1)
    return tuple(size // known if d == -1 else d for d in shape)


class ndarray:
    """A flat buffer plus a shape, with copy-on-index semantics."""

    # tinygrad spots a numpy array by str(type(x)) (helpers.is_numpy_ndarray),
    # so this class has to claim numpy's name to be accepted by Tensor().
    __module__ = "numpy"
    __slots__ = ("_buf", "dtype", "shape")

    def __init__(self, buf: _array.array, dt: _DType, shape: tuple[int, ...]):
        self._buf, self.dtype, self.shape = buf, dt, shape

    def __repr__(self) -> str:
        return f"ndarray(shape={self.shape}, dtype={self.dtype.name})"

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def size(self) -> int:
        return len(self._buf)

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def data(self) -> memoryview:
        """How tinygrad's NPY allocator copies out of an array it was handed."""
        return memoryview(self._buf)

    def tobytes(self) -> bytes:
        return self._buf.tobytes()

    def item(self):
        if self.size != 1:
            raise ValueError(f"item() on an array of size {self.size}")
        return self._buf[0]

    def copy(self) -> "ndarray":
        return ndarray(self._buf[:], self.dtype, self.shape)

    def reshape(self, *shape) -> "ndarray":
        """The one operation that shares storage; nothing writes through it."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return ndarray(self._buf, self.dtype, _resolve(shape, self.size))

    def astype(self, spec) -> "ndarray":
        dt = _dt(spec)
        if dt is self.dtype:
            return self.copy()
        from tinygrad import dtypes

        return _from_tensor(_to_tensor(self).cast(getattr(dtypes, dt.name)), self.shape)

    def _as(self, dt: _DType) -> "ndarray":
        return self if self.dtype is dt else self.astype(dt)

    # --- indexing -------------------------------------------------------

    def _span(self, key) -> tuple[int, int, int, bool]:
        """
        Flat (start, stop, step, is_scalar) for the indexes nammy actually uses:
        an integer per leading axis, then one slice or integer, with any
        remaining axes coming along whole. Anything else raises rather than
        being approximated, which is the point of a stand-in this small.
        """
        key = key if isinstance(key, tuple) else (key,)
        if len(key) > len(self.shape):
            raise IndexError(f"index {key} on shape {self.shape}")
        *leading, last = key
        base, width = 0, self.size
        for i, dim in zip(leading, self.shape):
            if not isinstance(i, int):
                raise NotImplementedError(f"only one axis may be sliced: {key}")
            width //= dim
            base += (i + dim if i < 0 else i) * width
        # `last` indexes the next axis; `block` is how many values each of its
        # entries covers, counting whatever axes were left unindexed.
        dim = self.shape[len(leading)]
        block = width // dim
        if isinstance(last, slice):
            start, stop, step = last.indices(dim)
            if step != 1 and block != 1:
                raise NotImplementedError(f"strided index on a non-final axis: {key}")
            return base + start * block, base + max(stop, start) * block, step, False
        last = last + dim if last < 0 else last
        return base + last * block, base + (last + 1) * block, 1, block == 1

    def __getitem__(self, key):
        start, stop, step, scalar = self._span(key)
        if scalar:
            return self._buf[start]
        values = self._buf[start:stop:step]
        return ndarray(values, self.dtype, (len(values),))

    def __setitem__(self, key, value) -> None:
        start, stop, step, scalar = self._span(key)
        if step != 1:
            raise NotImplementedError(f"assignment through a strided index: {key}")
        if scalar:
            self._buf[start] = value
        elif isinstance(value, ndarray):
            src = value._as(self.dtype)._buf
            if len(src) != stop - start:
                # array.array would silently resize; numpy would refuse.
                raise ValueError(f"cannot assign {len(src)} values to {stop - start}")
            self._buf[start:stop] = src
        else:
            self._buf[start:stop] = _array.array(self.dtype.code, [value]) * (stop - start)

    # --- arithmetic, all of it borrowed from tinygrad -------------------

    def _binary(self, other, op, swap: bool = False) -> "ndarray":
        a = _to_tensor(self)
        b = _to_tensor(other) if isinstance(other, ndarray) else other
        return _from_tensor(op(b, a) if swap else op(a, b), self.shape)

    def __add__(self, o):
        return self._binary(o, operator.add)

    def __sub__(self, o):
        return self._binary(o, operator.sub)

    def __mul__(self, o):
        return self._binary(o, operator.mul)

    def __truediv__(self, o):
        return self._binary(o, operator.truediv)

    def __pow__(self, o):
        return self._binary(o, operator.pow)

    def __or__(self, o):
        return self._binary(o, operator.or_)

    def __lshift__(self, o):
        return self._binary(o, operator.lshift)

    def __radd__(self, o):
        return self._binary(o, operator.add, swap=True)

    def __rsub__(self, o):
        return self._binary(o, operator.sub, swap=True)

    def __rmul__(self, o):
        return self._binary(o, operator.mul, swap=True)


# --- module-level constructors and reductions ---------------------------


def frombuffer(buffer, dtype=float64) -> ndarray:
    """Unlike numpy's, this copies; nothing in nammy relies on sharing."""
    dt = _dt(dtype)
    buf = _array.array(dt.code)
    buf.frombytes(buffer)
    return ndarray(buf, dt, (len(buf),))


def zeros(shape, dtype=float64) -> ndarray:
    dt = _dt(dtype)
    shape = shape if isinstance(shape, tuple) else (shape,)
    buf = _array.array(dt.code)
    buf.frombytes(bytes(math.prod(shape) * dt.itemsize))
    return ndarray(buf, dt, shape)


def empty(shape, dtype=float64) -> ndarray:
    # tinygrad's NPY allocator wants storage, not particular contents.
    return zeros(shape, dtype)


def concatenate(arrays) -> ndarray:
    arrays = list(arrays)
    dt = arrays[0].dtype
    buf = _array.array(dt.code)
    for a in arrays:
        buf.extend(a._as(dt)._buf)
    return ndarray(buf, dt, (len(buf),))


def stack(arrays) -> ndarray:
    arrays = list(arrays)
    if len({a.size for a in arrays}) > 1:
        raise ValueError("stack needs arrays of equal length")
    return concatenate(arrays).reshape(len(arrays), -1)


def clip(a, low, high) -> ndarray:
    return _from_tensor(_to_tensor(a).clip(low, high), a.shape)


def sum(a):
    """A plain sequence is reduced in Python; only an array is worth a kernel."""
    if not isinstance(a, ndarray):
        return math.fsum(a)
    return _to_tensor(a).sum().item()


def mean(a):
    if not isinstance(a, ndarray):
        return math.fsum(a) / len(a)
    return _to_tensor(a).mean().item()


def require(a, requirements=None) -> ndarray:
    """Everything this module produces is already C-contiguous."""
    return a


# tinygrad imports numpy by name in its own interop paths (Tensor(ndarray),
# Tensor.numpy, the NPY allocator), so the stand-in has to answer to that name
# as well. Nothing above imports tinygrad at module scope, so this is safe to
# do only once the module is fully built.
sys.modules.setdefault("numpy", sys.modules[__name__])
