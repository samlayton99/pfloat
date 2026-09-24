"""PArray: an n-dimensional array of values of a p-bit format, with correctly rounded arithmetic."""
from __future__ import annotations

import contextlib
import contextvars
import decimal
from fractions import Fraction
from math import inf, isinf, isnan, nan
import numbers

import numpy as np

from . import _lib
from ._formats import FP32, FP64, Format, as_format, round_rational

__all__ = ["PArray", "array", "asarray", "zeros", "ones", "full", "zeros_like", "ones_like", "full_like",
           "precision", "get_format", "set_format", "events", "sqrt", "tanh", "sum", "dot", "matmul",
           "round_to", "concatenate", "stack", "where", "maximum", "minimum", "absolute"]

_DEFAULT = contextvars.ContextVar("pbit_format", default=None)
_OPS = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3}


# ---------------------------------------------------------------- the working format

def get_format() -> Format | None:
    """The default format used when none is given (None unless set)."""
    return _DEFAULT.get()


def set_format(fmt) -> None:
    """Set the default format (a Format, an integer p, or a preset name; None to clear)."""
    _DEFAULT.set(None if fmt is None else as_format(fmt))


@contextlib.contextmanager
def precision(fmt):
    """Context manager: ``with pbit.precision(24): ...`` makes the format the default inside."""
    token = _DEFAULT.set(as_format(fmt))
    try:
        yield _DEFAULT.get()
    finally:
        _DEFAULT.reset(token)


def _resolve(fmt) -> Format:
    if fmt is not None:
        return as_format(fmt)
    default = _DEFAULT.get()
    if default is None:
        raise TypeError("no format given: pass fmt=... (a pbit.Format, an integer p, or a preset name) "
                        "or set one with pbit.precision(...) / pbit.set_format(...)")
    return default


def _kernel(fmt: Format):
    """Native binary64/binary32 kernels for exactly those formats (bit-identical), else the emulator."""
    if fmt == FP64:
        return _lib.library("f64")
    if fmt == FP32:
        return _lib.library("f32")
    return _lib.library("emul")


def events() -> dict:
    """Counts since the last call (then reset) from the emulator: results that were NaN or
    infinite, overflows, results in the subnormal range or rounded to zero, and inputs that were
    not format values (always 0 through the public API)."""
    import ctypes
    buf = (ctypes.c_long * 4)()
    _lib.library("emul").pb_events(buf)
    return {"nonfinite": buf[0], "not_format_input": buf[1], "overflow": buf[2], "underflow": buf[3]}


# ---------------------------------------------------------------- conversion into a format

def _c(a) -> np.ndarray:
    """A C-contiguous float64 copy that keeps the shape (np.ascontiguousarray turns 0-d into 1-d)."""
    return np.array(a, dtype=np.float64, order="C", copy=True)


def round_to(values, fmt) -> np.ndarray:
    """Round binary64 values into the format (nearest, ties to even); returns a float64 ndarray."""
    fmt = as_format(fmt)
    v = _c(values)
    out = np.empty_like(v)
    if _lib.library("emul").pb_round(_lib.ptr(fmt.constants()), v.size, _lib.ptr(v), _lib.ptr(out)):
        raise ValueError(f"unsupported format {fmt!r}")
    return out


def _exact(v):
    """An exact rational (or a float to be rounded, or a special value) for one element."""
    if isinstance(v, (bool, np.bool_)):
        return Fraction(int(v))
    if isinstance(v, float):
        return v
    if isinstance(v, np.floating):
        if v.dtype.itemsize <= 8 or not np.isfinite(v):
            return float(v)
        return Fraction(*v.as_integer_ratio())  # extended precision (longdouble)
    if isinstance(v, numbers.Rational):
        return Fraction(v)
    if isinstance(v, decimal.Decimal):
        return float(v) if not v.is_finite() else Fraction(v)
    if isinstance(v, PArray):
        return float(v)
    name = type(v).__module__.split(".")[0]
    if name == "mpmath":
        import mpmath
        if mpmath.isnan(v) or mpmath.isinf(v):
            return float(v)
        sign, man, exp, _ = v._mpf_
        return (-1) ** sign * Fraction(man) * Fraction(2) ** exp
    if name == "gmpy2":
        import gmpy2
        if gmpy2.is_nan(v) or gmpy2.is_infinite(v):
            return float(v)
        return Fraction(*v.as_integer_ratio())
    if isinstance(v, numbers.Real):
        return float(v)
    raise TypeError(f"cannot convert {type(v).__name__} to a p-bit value")


def _from_object(arr: np.ndarray, fmt: Format) -> np.ndarray:
    out = np.empty(arr.shape, dtype=np.float64)
    flat_in, flat_out = arr.reshape(-1), out.reshape(-1)
    floats = []
    for i, v in enumerate(flat_in):
        e = _exact(v)
        if isinstance(e, float):
            floats.append(i)
            flat_out[i] = e
        else:
            flat_out[i] = round_rational(e, fmt)
    if floats:
        idx = np.array(floats)
        flat_out[idx] = round_to(flat_out[idx], fmt)
    return out


def _is_torch(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "torch" and hasattr(obj, "detach")


def array(obj, fmt=None) -> "PArray":
    """Convert data into a PArray of the format, rounding each value correctly (nearest, ties to
    even). Accepts PArrays (re-rounded if the format differs), numpy arrays and scalars of any
    real dtype, Python numbers and nested lists, torch tensors, and exact numbers (int, Fraction,
    Decimal, mpmath, gmpy2), which are rounded from their exact values. Always copies."""
    fmt = _resolve(fmt)
    if isinstance(obj, PArray):
        return obj.astype(fmt)
    if _is_torch(obj):
        t = obj.detach().cpu()
        obj = t.double().numpy() if t.is_floating_point() else t.numpy()
    arr = np.asarray(obj)
    kind = arr.dtype.kind
    if kind == "c":
        raise TypeError("complex values are not supported")
    if kind == "f" and arr.dtype.itemsize <= 8:
        return PArray._wrap(round_to(arr.astype(np.float64), fmt), fmt)
    if kind in "iu" and arr.size and np.max(np.abs(arr.astype(object))) > 2 ** 53:
        return PArray._wrap(_from_object(arr.astype(object), fmt), fmt)
    if kind in "iub":
        return PArray._wrap(round_to(arr.astype(np.float64), fmt), fmt)
    return PArray._wrap(_from_object(arr.astype(object), fmt), fmt)


def asarray(obj, fmt=None) -> "PArray":
    """Like array(), but returns obj itself when it is already a PArray of the format."""
    if isinstance(obj, PArray) and (fmt is None or as_format(fmt) == obj.fmt):
        return obj
    return array(obj, fmt)


def full(shape, value, fmt=None) -> "PArray":
    fmt = _resolve(fmt)
    v = array(value, fmt)
    return PArray._wrap(np.full(shape, float(v)), fmt)


def zeros(shape, fmt=None) -> "PArray":
    return full(shape, 0, fmt)


def ones(shape, fmt=None) -> "PArray":
    return full(shape, 1, fmt)


def zeros_like(x: "PArray") -> "PArray":
    return PArray._wrap(np.zeros(x.shape), x.fmt)


def ones_like(x: "PArray") -> "PArray":
    return PArray._wrap(np.ones(x.shape), x.fmt)


def full_like(x: "PArray", value) -> "PArray":
    return full(x.shape, value, x.fmt)


# ---------------------------------------------------------------- the array type

def _pair(x, y):
    """Two operands as PArrays of one format; plain numbers and arrays are rounded into it."""
    if isinstance(x, PArray) and isinstance(y, PArray):
        if x.fmt != y.fmt:
            raise TypeError(f"mixed formats {x.fmt!r} and {y.fmt!r}: convert one with .astype(...)")
        return x, y
    if isinstance(x, PArray):
        return x, array(y, x.fmt)
    return array(x, y.fmt), y


def _binary(op: str, x, y) -> "PArray":
    x, y = _pair(x, y)
    a, b = np.broadcast_arrays(x._v, y._v)
    a, b = _c(a), _c(b)
    out = np.empty(a.shape)
    _kernel(x.fmt).pb_binary(_lib.ptr(x.fmt.constants()), _OPS[op], a.size, _lib.ptr(a), _lib.ptr(b), _lib.ptr(out))
    return PArray._wrap(out, x.fmt)


def _unary(code: int, x) -> "PArray":
    x = x if isinstance(x, PArray) else array(x)
    a = _c(x._v)
    out = np.empty(a.shape)
    _kernel(x.fmt).pb_unary(_lib.ptr(x.fmt.constants()), code, a.size, _lib.ptr(a), _lib.ptr(out))
    return PArray._wrap(out, x.fmt)


def sqrt(x) -> "PArray":
    """Correctly rounded square root."""
    return _unary(0, x)


def tanh(x) -> "PArray":
    """tanh computed in the format from +, -, *, / only (at most 3 ulp; no higher-precision
    library call): tanh|z| = -E/(E + 2), E = expm1(-2|z|) by Cody-Waite reduction and a Taylor
    polynomial of degree set by p."""
    return _unary(1, x)


def sum(x, axis=None, keepdims=False) -> "PArray":  # noqa: A001
    """Sum left to right along the axis (all elements, in C order, when axis is None)."""
    x = x if isinstance(x, PArray) else array(x)
    if axis is None:
        v = _c(x._v.reshape(1, -1))
        out = PArray._wrap(_row_sums(v, x.fmt).reshape(()), x.fmt)
        return out.reshape((1,) * x.ndim) if keepdims else out
    axis = axis % x.ndim if x.ndim else 0
    moved = _c(np.moveaxis(x._v, axis, -1))
    lead = moved.shape[:-1]
    sums = _row_sums(moved.reshape(-1, moved.shape[-1]), x.fmt).reshape(lead)
    out = PArray._wrap(sums, x.fmt)
    return PArray._wrap(np.expand_dims(sums, axis), x.fmt) if keepdims else out


def _row_sums(rows: np.ndarray, fmt: Format) -> np.ndarray:
    rows = _c(rows)
    out = np.empty(rows.shape[0])
    _kernel(fmt).pb_sum(_lib.ptr(fmt.constants()), rows.shape[0], rows.shape[1], _lib.ptr(rows), _lib.ptr(out))
    return out


def matmul(x, y) -> "PArray":
    """Matrix product; each entry is a sum of products accumulated in order from zero, the
    operation order of reference BLAS DGEMM. 1-D operands are treated as vectors."""
    x, y = _pair(x, y)
    if x.ndim == 0 or y.ndim == 0:
        raise ValueError("matmul: scalar operands are not allowed, use *")
    a = x._v if x.ndim > 1 else x._v.reshape(1, -1)
    b = y._v if y.ndim > 1 else y._v.reshape(-1, 1)
    if a.ndim > 2 or b.ndim > 2:
        raise ValueError("matmul supports 1-D and 2-D operands")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul: shapes {x.shape} and {y.shape} are not aligned")
    a, b = _c(a), _c(b)
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    out = np.empty((m, n))
    _kernel(x.fmt).pb_matmul(_lib.ptr(x.fmt.constants()), m, k, n, _lib.ptr(a), _lib.ptr(b), _lib.ptr(out))
    if x.ndim == 1 and y.ndim == 1:
        out = out.reshape(())
    elif x.ndim == 1:
        out = out.reshape(n)
    elif y.ndim == 1:
        out = out.reshape(m)
    return PArray._wrap(out, x.fmt)


def dot(x, y) -> "PArray":
    """Dot product (1-D), matrix-vector or matrix product (2-D), as matmul."""
    return matmul(x, y)


def _exact_select(fn, *args):
    xs = [a for a in args if isinstance(a, PArray)]
    fmt = xs[0].fmt
    vals = [a._v if isinstance(a, PArray) else array(a, fmt)._v for a in args]
    if any(isinstance(a, PArray) and a.fmt != fmt for a in args):
        raise TypeError("mixed formats")
    return PArray._wrap(np.asarray(fn(*vals), dtype=np.float64), fmt)


def maximum(x, y) -> "PArray":
    """Elementwise maximum (exact: selects one of the operands)."""
    return _exact_select(np.maximum, x, y)


def minimum(x, y) -> "PArray":
    return _exact_select(np.minimum, x, y)


def absolute(x) -> "PArray":
    return _exact_select(np.abs, x)


def where(cond, x, y) -> "PArray":
    """Select elementwise from x or y (exact)."""
    return _exact_select(lambda a, b: np.where(np.asarray(cond), a, b), x, y)


def concatenate(arrays, axis=0) -> "PArray":
    fmt = arrays[0].fmt
    if not all(isinstance(a, PArray) and a.fmt == fmt for a in arrays):
        raise TypeError("concatenate: all arrays must be PArrays of one format")
    return PArray._wrap(np.concatenate([a._v for a in arrays], axis=axis), fmt)


def stack(arrays, axis=0) -> "PArray":
    fmt = arrays[0].fmt
    if not all(isinstance(a, PArray) and a.fmt == fmt for a in arrays):
        raise TypeError("stack: all arrays must be PArrays of one format")
    return PArray._wrap(np.stack([a._v for a in arrays], axis=axis), fmt)


class PArray:
    """An array whose every value is a value of one p-bit format (``.fmt``).

    Arithmetic (+, -, *, /, unary -, abs, ``@``, sqrt, tanh, sum, dot) is correctly rounded in the
    format; operands that are plain numbers or numpy arrays are first rounded into it. Operations
    between different formats raise; convert with ``.astype(fmt)``. numpy functions and ufuncs
    that pbit does not implement raise TypeError rather than silently computing in binary64; leave
    the format explicitly with ``.to_numpy()``.

    Conversions out: ``.to_numpy()`` (binary64, exact), ``.to_numpy(np.float32)`` (only if exactly
    representable), ``.to_torch()``, ``.tolist()``, ``.to_fractions()``, ``float(x)``.
    """

    __slots__ = ("_v", "fmt")
    __array_priority__ = 1000
    __hash__ = None

    def __init__(self, data, fmt=None):
        other = array(data, fmt)
        self._v, self.fmt = other._v, other.fmt

    @classmethod
    def _wrap(cls, values: np.ndarray, fmt: Format) -> "PArray":
        obj = cls.__new__(cls)
        obj._v = np.asarray(values, dtype=np.float64)
        obj.fmt = fmt
        return obj

    # ---- basic properties and views
    @property
    def values(self) -> np.ndarray:
        """The values as a read-only binary64 view (exact)."""
        v = self._v.view()
        v.setflags(write=False)
        return v

    shape = property(lambda self: self._v.shape)
    ndim = property(lambda self: self._v.ndim)
    size = property(lambda self: self._v.size)
    T = property(lambda self: PArray._wrap(self._v.T, self.fmt))

    def __len__(self):
        return len(self._v)

    def __iter__(self):
        for v in self._v:
            yield PArray._wrap(v, self.fmt)

    def __getitem__(self, key):
        return PArray._wrap(self._v[key], self.fmt)

    def __setitem__(self, key, value):
        self._v[key] = array(value, self.fmt)._v

    def reshape(self, *shape):
        return PArray._wrap(self._v.reshape(*shape), self.fmt)

    def transpose(self, *axes):
        return PArray._wrap(self._v.transpose(*axes), self.fmt)

    def ravel(self):
        return PArray._wrap(self._v.ravel(), self.fmt)

    def flatten(self):
        return PArray._wrap(self._v.flatten(), self.fmt)

    def squeeze(self, axis=None):
        return PArray._wrap(self._v.squeeze(axis), self.fmt)

    def copy(self):
        return PArray._wrap(self._v.copy(), self.fmt)

    def astype(self, fmt) -> "PArray":
        """Round into another format (a copy)."""
        fmt = as_format(fmt)
        if fmt == self.fmt:
            return self.copy()
        return PArray._wrap(round_to(self._v, fmt), fmt)

    # ---- conversions out
    def to_numpy(self, dtype=np.float64) -> np.ndarray:
        """A numpy copy. binary64 is always exact; a narrower float dtype is allowed only when
        every value is exactly representable in it."""
        dtype = np.dtype(dtype)
        if dtype == np.float64:
            return self._v.copy()
        if dtype.kind != "f":
            raise TypeError("to_numpy supports floating dtypes")
        out = self._v.astype(dtype)
        back = out.astype(np.float64)
        same = (back == self._v) | (np.isnan(back) & np.isnan(self._v))
        if not np.all(same):
            raise ValueError(f"values of {self.fmt!r} are not all exactly representable as {dtype}; "
                             "round them first, e.g. .astype(pbit.FP32).to_numpy(np.float32)")
        return out

    def __array__(self, dtype=None, copy=None):
        return self.to_numpy(dtype if dtype is not None else np.float64)

    def to_torch(self, dtype=None, device=None):
        """A torch tensor (float64 by default, exact); a narrower dtype only if exact."""
        import torch
        t = torch.from_numpy(self._v.copy())
        if dtype is not None and dtype != torch.float64:
            out = t.to(dtype)
            if not torch.equal(out.double(), t) and not bool(torch.isnan(t).any()):
                raise ValueError(f"values of {self.fmt!r} are not all exactly representable as {dtype}")
            t = out
        return t.to(device) if device is not None else t

    def tolist(self):
        return self._v.tolist()

    def to_fractions(self) -> np.ndarray:
        """An object array of exact Fractions (NaN and infinities stay floats)."""
        return np.vectorize(lambda v: Fraction(v) if np.isfinite(v) else v, otypes=[object])(self._v)

    def __float__(self):
        return float(self._v)

    def __int__(self):
        return int(self._v)

    def __bool__(self):
        return bool(self._v)

    def __repr__(self):
        body = np.array2string(self._v, separator=", ", floatmode="unique", threshold=200)
        return f"PArray({body}, fmt={self.fmt!r})"

    __str__ = __repr__

    # ---- arithmetic
    def __add__(self, o): return _binary("add", self, o)
    def __radd__(self, o): return _binary("add", o, self)
    def __sub__(self, o): return _binary("subtract", self, o)
    def __rsub__(self, o): return _binary("subtract", o, self)
    def __mul__(self, o): return _binary("multiply", self, o)
    def __rmul__(self, o): return _binary("multiply", o, self)
    def __truediv__(self, o): return _binary("divide", self, o)
    def __rtruediv__(self, o): return _binary("divide", o, self)
    def __matmul__(self, o): return matmul(self, o)
    def __rmatmul__(self, o): return matmul(o, self)
    def __neg__(self): return PArray._wrap(-self._v, self.fmt)
    def __pos__(self): return self.copy()
    def __abs__(self): return PArray._wrap(np.abs(self._v), self.fmt)

    def __pow__(self, k):
        if k == 2:
            return self * self
        if k == 0.5:
            return sqrt(self)
        raise TypeError("PArray supports x**2 and x**0.5 only; other powers have no single correctly "
                        "rounded definition here")

    def __eq__(self, o): return self._v == _cmp(self, o)
    def __ne__(self, o): return self._v != _cmp(self, o)
    def __lt__(self, o): return self._v < _cmp(self, o)
    def __le__(self, o): return self._v <= _cmp(self, o)
    def __gt__(self, o): return self._v > _cmp(self, o)
    def __ge__(self, o): return self._v >= _cmp(self, o)

    # ---- reductions and products
    def sum(self, axis=None, keepdims=False):
        return sum(self, axis=axis, keepdims=keepdims)

    def dot(self, o):
        return matmul(self, o)

    def max(self, axis=None):
        return PArray._wrap(np.max(self._v, axis=axis), self.fmt)

    def min(self, axis=None):
        return PArray._wrap(np.min(self._v, axis=axis), self.fmt)

    # ---- numpy protocol: implemented operations run in the format, the rest raise
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if kwargs.get("out") is not None:
            raise TypeError("pbit: out= is not supported")
        if method == "reduce" and ufunc is np.add:
            (x,) = inputs
            return sum(x, axis=kwargs.get("axis", 0), keepdims=kwargs.get("keepdims", False))
        if method != "__call__" or kwargs:
            raise TypeError(f"pbit: numpy {ufunc.__name__}.{method} is not implemented at p-bit precision")
        name = ufunc.__name__
        if name in ("add", "subtract", "multiply"):
            return _binary(name, *inputs)
        if name in ("true_divide", "divide"):
            return _binary("divide", *inputs)
        if name == "matmul":
            return matmul(*inputs)
        if name == "sqrt":
            return sqrt(inputs[0])
        if name == "tanh":
            return tanh(inputs[0])
        if name == "square":
            return inputs[0] * inputs[0]
        if name in ("negative", "positive", "absolute", "fabs", "floor", "ceil", "trunc", "rint",
                    "maximum", "minimum", "fmax", "fmin", "copysign"):
            return _exact_select(getattr(np, name), *inputs)
        if name in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal", "isnan",
                    "isinf", "isfinite", "signbit"):
            vals = [a._v if isinstance(a, PArray) else a for a in inputs]
            return ufunc(*vals)
        raise TypeError(f"pbit: numpy ufunc '{name}' is not implemented at p-bit precision. Convert "
                        "explicitly with .to_numpy() if you intend to leave the format.")

    def __array_function__(self, func, types, args, kwargs):
        from . import linalg
        handled = {
            np.sum: sum, np.dot: dot, np.matmul: matmul,
            np.maximum: maximum, np.minimum: minimum, np.abs: absolute, np.absolute: absolute,
            np.where: where, np.concatenate: concatenate, np.stack: stack,
            np.linalg.lstsq: linalg.lstsq,
            np.transpose: lambda a, axes=None: a.transpose(*(axes or ())),
            np.reshape: lambda a, shape, *r, **k: a.reshape(shape),
            np.ravel: lambda a, *r, **k: a.ravel(),
            np.copy: lambda a, *r, **k: a.copy(),
            np.shape: lambda a: a.shape, np.ndim: lambda a: a.ndim, np.size: lambda a, *r: a.size,
            np.zeros_like: lambda a, *r, **k: zeros_like(a), np.ones_like: lambda a, *r, **k: ones_like(a),
            np.array_equal: lambda a, b, *r, **k: bool(np.array_equal(np.asarray(a), np.asarray(b))),
        }
        if func in handled:
            return handled[func](*args, **kwargs)
        raise TypeError(f"pbit: numpy.{func.__name__} is not implemented at p-bit precision. Convert "
                        "explicitly with .to_numpy() if you intend to leave the format.")


def _cmp(x: PArray, o):
    return o._v if isinstance(o, PArray) else np.asarray(o, dtype=np.float64)
