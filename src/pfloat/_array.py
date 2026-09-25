"""PArray: an n-dimensional array of values of a p-bit format, with correctly rounded arithmetic."""
from __future__ import annotations

import contextlib
import contextvars
import ctypes
import decimal
from fractions import Fraction
from functools import lru_cache
from math import ceil, log10
import numbers

import numpy as np

from . import _lib, _mp
from ._formats import FP32, FP64, Format, as_format, exact_constants, round_fraction

__all__ = ["PArray", "array", "asarray", "zeros", "ones", "full", "zeros_like", "ones_like", "full_like",
           "precision", "get_format", "set_format", "events", "sqrt", "tanh", "sum", "dot", "matmul",
           "round_to", "concatenate", "stack", "where", "maximum", "minimum", "absolute"]

_DEFAULT = contextvars.ContextVar("pfloat_format", default=None)
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
    """Context manager: ``with pfloat.precision(24): ...`` makes the format the default inside."""
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
        raise TypeError("no format given: pass fmt=... (a pfloat.Format, an integer p, or a preset name) "
                        "or set one with pfloat.precision(...) / pfloat.set_format(...)")
    return default


def events() -> dict:
    """Counts since the last call (then reset) from the p <= 53 emulator: results that were NaN or
    infinite, overflows, results in the subnormal range or rounded to zero, and inputs that were
    not format values (always 0 through the public API)."""
    buf = (ctypes.c_long * 4)()
    _lib.library("emul").pb_events(buf)
    return {"nonfinite": buf[0], "not_format_input": buf[1], "overflow": buf[2], "underflow": buf[3]}


# ---------------------------------------------------------------- storage and kernels

def _storage(fmt: Format) -> np.dtype:
    """binary64 for p <= 53 (the values are carried exactly in doubles); MPFR records above."""
    return _mp.dtype(_mp.limbs_for(fmt.p)) if fmt.extended else np.dtype(np.float64)


def _kernel(fmt: Format):
    """Native binary64/binary32 kernels for exactly those formats (bit-identical to the emulator),
    the emulator for other p <= 53, the MPFR build above."""
    if fmt.extended:
        return _mp.library(_mp.limbs_for(fmt.p))
    if fmt == FP64:
        return _lib.library("f64")
    if fmt == FP32:
        return _lib.library("f32")
    return _lib.library("emul")


@lru_cache(maxsize=None)
def _ext_constants(fmt: Format) -> np.ndarray:
    values, coeffs = exact_constants(fmt.p, fmt.emin, fmt.emax)
    entries = [fmt.p, fmt.emin, fmt.emax, *values, len(coeffs), *coeffs]
    return _mp.encode([Fraction(v) for v in entries], fmt.p, _mp.limbs_for(fmt.p))


def _ptr(fmt: Format, arr: np.ndarray):
    return _mp.vptr(arr) if fmt.extended else _lib.ptr(arr)


def _cargs(fmt: Format):
    """The two constants arguments of every kernel: doubles, and values of the format."""
    c = fmt.constants()
    if fmt.extended:
        return _lib.ptr(c), _mp.vptr(_ext_constants(fmt))
    return _lib.ptr(c), _lib.ptr(c)


def _c(a) -> np.ndarray:
    """A C-contiguous copy that keeps shape and dtype (np.ascontiguousarray turns 0-d into 1-d)."""
    return np.array(a, order="C", copy=True)


def _empty(shape, fmt: Format) -> np.ndarray:
    return np.empty(shape, dtype=_storage(fmt))


# ---------------------------------------------------------------- conversion into a format

def round_to(values, fmt) -> np.ndarray:
    """Round binary64 values into the format (nearest, ties to even). Returns the stored values:
    float64 for p <= 53, MPFR records above."""
    fmt = as_format(fmt)
    v = np.array(values, dtype=np.float64, order="C", copy=True)
    if fmt.extended:
        out = _empty(v.shape, fmt)
        if _kernel(fmt).pb_from_double(_lib.ptr(fmt.constants()), v.size, _lib.ptr(v), _mp.vptr(out)):
            raise ValueError(f"unsupported format {fmt!r}")
        return out
    out = np.empty_like(v)
    if _lib.library("emul").pb_round(_lib.ptr(fmt.constants()), v.size, _lib.ptr(v), _lib.ptr(out)):
        raise ValueError(f"unsupported format {fmt!r}")
    return out


def _exact(v):
    """One element as an exact rational, or a float (to be rounded, or a special value)."""
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


def _rounded_exact(e, fmt: Format):
    """The exactly rounded value of one element: Fraction, or float for zeros/infinities/NaN."""
    if isinstance(e, float):
        if e != e or e in (float("inf"), float("-inf")) or e == 0:
            return e
        e = Fraction(e)
    return round_fraction(e, fmt)


def _from_exact_values(flat: list, shape, fmt: Format) -> np.ndarray:
    """Store already-rounded values (Fractions and special floats)."""
    if fmt.extended:
        return _mp.encode(flat, fmt.p, _mp.limbs_for(fmt.p)).reshape(shape)
    return np.array([float(v) for v in flat], dtype=np.float64).reshape(shape)


def _from_object(arr: np.ndarray, fmt: Format) -> np.ndarray:
    flat = [_rounded_exact(_exact(v), fmt) for v in arr.reshape(-1)]
    return _from_exact_values(flat, arr.shape, fmt)


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
    if kind == "V":
        raise TypeError("structured arrays are not values; build extended values with pfloat.array(...)")
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
    v = array(value, fmt)._v
    out = _empty(shape, fmt)
    out[...] = v
    return PArray._wrap(out, fmt)


def zeros(shape, fmt=None) -> "PArray":
    return full(shape, 0, fmt)


def ones(shape, fmt=None) -> "PArray":
    return full(shape, 1, fmt)


def zeros_like(x: "PArray") -> "PArray":
    return full(x.shape, 0, x.fmt)


def ones_like(x: "PArray") -> "PArray":
    return full(x.shape, 1, x.fmt)


def full_like(x: "PArray", value) -> "PArray":
    return full(x.shape, value, x.fmt)


# ---------------------------------------------------------------- arithmetic

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
    fmt = x.fmt
    a, b = np.broadcast_arrays(x._v, y._v)
    a, b = _c(a), _c(b)
    out = _empty(a.shape, fmt)
    _kernel(fmt).pb_binary(*_cargs(fmt), _OPS[op], a.size, _ptr(fmt, a), _ptr(fmt, b), _ptr(fmt, out))
    return PArray._wrap(out, fmt)


def _unary(code: int, x) -> "PArray":
    x = x if isinstance(x, PArray) else array(x)
    fmt = x.fmt
    a = _c(x._v)
    out = _empty(a.shape, fmt)
    _kernel(fmt).pb_unary(*_cargs(fmt), code, a.size, _ptr(fmt, a), _ptr(fmt, out))
    return PArray._wrap(out, fmt)


def sqrt(x) -> "PArray":
    """Correctly rounded square root."""
    return _unary(0, x)


def tanh(x) -> "PArray":
    """tanh computed in the format from +, -, *, / only (at most 3 ulp; no higher-precision
    library call): tanh|z| = -E/(E + 2), E = expm1(-2|z|) by Cody-Waite reduction and a Taylor
    polynomial of degree set by p."""
    return _unary(1, x)


def _row_sums(rows: np.ndarray, fmt: Format) -> np.ndarray:
    rows = _c(rows)
    out = _empty(rows.shape[0], fmt)
    _kernel(fmt).pb_sum(*_cargs(fmt), rows.shape[0], rows.shape[1], _ptr(fmt, rows), _ptr(fmt, out))
    return out


def sum(x, axis=None, keepdims=False) -> "PArray":  # noqa: A001
    """Sum left to right along the axis (all elements, in C order, when axis is None)."""
    x = x if isinstance(x, PArray) else array(x)
    if axis is None:
        total = _row_sums(x._v.reshape(1, -1), x.fmt).reshape(())
        out = PArray._wrap(total, x.fmt)
        return out.reshape((1,) * x.ndim) if keepdims else out
    axis = axis % x.ndim if x.ndim else 0
    moved = np.moveaxis(x._v, axis, -1)
    lead = moved.shape[:-1]
    sums = _row_sums(moved.reshape(-1, moved.shape[-1]), x.fmt).reshape(lead)
    return PArray._wrap(np.expand_dims(sums, axis) if keepdims else sums, x.fmt)


def matmul(x, y) -> "PArray":
    """Matrix product; each entry is a sum of products accumulated in order from zero, the
    operation order of reference BLAS DGEMM. 1-D operands are treated as vectors."""
    x, y = _pair(x, y)
    fmt = x.fmt
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
    out = _empty((m, n), fmt)
    _kernel(fmt).pb_matmul(*_cargs(fmt), m, k, n, _ptr(fmt, a), _ptr(fmt, b), _ptr(fmt, out))
    if x.ndim == 1 and y.ndim == 1:
        out = out.reshape(())
    elif x.ndim == 1:
        out = out.reshape(n)
    elif y.ndim == 1:
        out = out.reshape(m)
    return PArray._wrap(out, fmt)


def dot(x, y) -> "PArray":
    """Dot product (1-D), matrix-vector or matrix product (2-D), as matmul."""
    return matmul(x, y)


# ---------------------------------------------------------------- exact operations

_CMP = {"less": 0, "less_equal": 1, "equal": 2, "not_equal": 3, "greater": 4, "greater_equal": 5}


def _compare(name: str, x, y) -> np.ndarray:
    if not isinstance(x, PArray) and not isinstance(y, PArray):
        raise TypeError("compare needs a PArray")
    x, y = _pair(x, y)
    fmt = x.fmt
    if not fmt.extended:
        return getattr(np, name)(x._v, y._v)
    a, b = np.broadcast_arrays(x._v, y._v)
    a, b = _c(a), _c(b)
    out = np.empty(a.shape, dtype=np.int8)
    _kernel(fmt).pb_compare(fmt.constants().ctypes.data_as(ctypes.POINTER(ctypes.c_double)), _CMP[name],
                            a.size, _mp.vptr(a), _mp.vptr(b), _mp.vptr(out))
    return out.astype(bool)


def _flags(x: "PArray", what: str) -> np.ndarray:
    if not x.fmt.extended:
        return getattr(np, what)(x._v)
    kind, sign = x._v["kind"], x._v["sign"]
    return {"isnan": kind == 3, "isinf": kind == 2, "isfinite": kind < 2, "signbit": (sign == 1) & (kind != 3)}[what]


def _with_sign(x: "PArray", mode: str) -> "PArray":
    """Exact sign operations: 'negative', 'absolute' (NaN left alone)."""
    if not x.fmt.extended:
        return PArray._wrap(np.negative(x._v) if mode == "negative" else np.abs(x._v), x.fmt)
    v = x._v.copy()
    real = v["kind"] != 3
    v["sign"] = np.where(real, 1 - v["sign"], v["sign"]) if mode == "negative" else np.where(real, 0, v["sign"])
    return PArray._wrap(v, x.fmt)


def maximum(x, y) -> "PArray":
    """Elementwise maximum (exact: selects one of the operands; NaN propagates)."""
    x, y = _pair(x, y)
    a, b = np.broadcast_arrays(x._v, y._v)
    pick_b = _compare("greater", PArray._wrap(b, x.fmt), PArray._wrap(a, x.fmt)) | _flags(PArray._wrap(b, x.fmt), "isnan")
    return PArray._wrap(np.where(pick_b, b, a), x.fmt)


def minimum(x, y) -> "PArray":
    x, y = _pair(x, y)
    a, b = np.broadcast_arrays(x._v, y._v)
    pick_b = _compare("less", PArray._wrap(b, x.fmt), PArray._wrap(a, x.fmt)) | _flags(PArray._wrap(b, x.fmt), "isnan")
    return PArray._wrap(np.where(pick_b, b, a), x.fmt)


def absolute(x) -> "PArray":
    return _with_sign(x if isinstance(x, PArray) else array(x), "absolute")


def where(cond, x, y) -> "PArray":
    """Select elementwise from x or y (exact)."""
    x, y = _pair(x, y)
    return PArray._wrap(np.where(np.asarray(cond), x._v, y._v), x.fmt)


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


def _reduce_select(x: "PArray", axis, fn) -> "PArray":
    if not x.fmt.extended:
        return PArray._wrap(getattr(np, fn)(x._v, axis=axis), x.fmt)
    v = x._v.reshape(-1) if axis is None else np.moveaxis(x._v, axis, 0)
    acc = PArray._wrap(v[0], x.fmt)
    for k in range(1, v.shape[0]):
        acc = (maximum if fn == "max" else minimum)(acc, PArray._wrap(v[k], x.fmt))
    return acc


class PArray:
    """An array whose every value is a value of one p-bit format (``.fmt``).

    Arithmetic (+, -, *, /, unary -, abs, ``@``, sqrt, tanh, sum, dot) is correctly rounded in the
    format; operands that are plain numbers or numpy arrays are first rounded into it. Operations
    between different formats raise; convert with ``.astype(fmt)``. numpy functions and ufuncs
    that pfloat does not implement raise TypeError rather than silently computing in binary64;
    leave the format explicitly with ``.to_numpy()``.

    Conversions out: ``.to_numpy()`` (binary64; exact, or an error for p > 53 values that do not
    fit unless ``rounding=True``), ``.to_torch()``, ``.to_fractions()``, ``.to_mpmath()``,
    ``.tolist()``, ``float(x)``.
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
        obj._v = np.asarray(values, dtype=_storage(fmt))
        obj.fmt = fmt
        return obj

    # ---- basic properties and views
    @property
    def values(self) -> np.ndarray:
        """The stored values as a read-only view: float64 (exact) for p <= 53, MPFR records above."""
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
        if not self.fmt.extended:
            return PArray._wrap(round_to(self._v, fmt), fmt)
        flat = [_rounded_exact(v if not isinstance(v, float) else v, fmt) for v in self._exact_flat()]
        return PArray._wrap(_from_exact_values(flat, self.shape, fmt), fmt)

    # ---- conversions out
    def _exact_flat(self) -> list:
        if self.fmt.extended:
            return _mp.decode(self._v, self.fmt.p)
        return [v if not np.isfinite(v) or v == 0 else Fraction(v) for v in self._v.reshape(-1).tolist()]

    def to_numpy(self, dtype=np.float64, rounding: bool = False) -> np.ndarray:
        """A numpy copy. For p <= 53, binary64 is always exact. A narrower float dtype, or binary64
        for p > 53, is allowed only when every value is exactly representable, unless
        rounding=True (round to nearest)."""
        dtype = np.dtype(dtype)
        if dtype.kind != "f":
            raise TypeError("to_numpy supports floating dtypes")
        if self.fmt.extended:
            v = _c(self._v)
            as64 = np.empty(v.shape)
            inexact = ctypes.c_long(0)
            _kernel(self.fmt).pb_to_double(_lib.ptr(self.fmt.constants()), v.size, _mp.vptr(v), _lib.ptr(as64),
                                           ctypes.byref(inexact))
            if inexact.value and not rounding:
                raise ValueError(f"{inexact.value} values of {self.fmt!r} are not exactly representable in binary64; "
                                 "pass rounding=True to round them, or use .to_fractions() / .to_mpmath()")
        else:
            as64 = self._v.copy()
        if dtype == np.float64:
            return as64
        out = as64.astype(dtype)
        back = out.astype(np.float64)
        same = (back == as64) | (np.isnan(back) & np.isnan(as64))
        if not np.all(same) and not rounding:
            raise ValueError(f"values of {self.fmt!r} are not all exactly representable as {dtype}; round them "
                             "first (.astype(...)) or pass rounding=True")
        return out

    def __array__(self, dtype=None, copy=None):
        return self.to_numpy(dtype if dtype is not None else np.float64)

    def to_torch(self, dtype=None, device=None):
        """A torch tensor (float64 by default, exact); a narrower dtype only if exact."""
        import torch
        t = torch.from_numpy(self.to_numpy())
        if dtype is not None and dtype != torch.float64:
            out = t.to(dtype)
            if not torch.equal(out.double(), t) and not bool(torch.isnan(t).any()):
                raise ValueError(f"values of {self.fmt!r} are not all exactly representable as {dtype}")
            t = out
        return t.to(device) if device is not None else t

    def tolist(self):
        """Nested lists: floats for p <= 53 (exact); exact Fractions for p > 53."""
        if not self.fmt.extended:
            return self._v.tolist()
        return np.array(self._exact_flat(), dtype=object).reshape(self.shape).tolist()

    def to_fractions(self) -> np.ndarray:
        """An object array of exact Fractions (NaN and infinities stay floats)."""
        return np.array(self._exact_flat(), dtype=object).reshape(self.shape)

    def to_mpmath(self) -> np.ndarray:
        """An object array of mpmath numbers holding the values exactly."""
        import mpmath
        vals = [mpmath.mpf(v) if isinstance(v, float) else _mpf_exact(v) for v in self._exact_flat()]
        return np.array(vals, dtype=object).reshape(self.shape)

    def __float__(self):
        if not self.fmt.extended:
            return float(self._v)
        (v,) = self._exact_flat()
        return float(v)

    def __int__(self):
        return int(float(self))

    def __bool__(self):
        if self.size != 1:
            raise ValueError("the truth value of an array with more than one element is ambiguous")
        return bool(float(self))

    def __repr__(self):
        if not self.fmt.extended:
            body = np.array2string(self._v, separator=", ", floatmode="unique", threshold=200)
        else:
            import mpmath
            digits = int(ceil(self.fmt.p * log10(2))) + 1
            strs = [mpmath.nstr(_mpf_exact(v), digits) if not isinstance(v, float) else repr(v)
                    for v in self._exact_flat()]
            body = np.array2string(np.array(strs, dtype=object).reshape(self.shape), separator=", ",
                                   formatter={"all": str}, threshold=200)
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
    def __neg__(self): return _with_sign(self, "negative")
    def __pos__(self): return self.copy()
    def __abs__(self): return _with_sign(self, "absolute")

    def __pow__(self, k):
        if k == 2:
            return self * self
        if k == 0.5:
            return sqrt(self)
        raise TypeError("PArray supports x**2 and x**0.5 only; other powers have no single correctly "
                        "rounded definition here")

    def __eq__(self, o): return _compare("equal", self, o)
    def __ne__(self, o): return _compare("not_equal", self, o)
    def __lt__(self, o): return _compare("less", self, o)
    def __le__(self, o): return _compare("less_equal", self, o)
    def __gt__(self, o): return _compare("greater", self, o)
    def __ge__(self, o): return _compare("greater_equal", self, o)

    # ---- reductions and products
    def sum(self, axis=None, keepdims=False):
        return sum(self, axis=axis, keepdims=keepdims)

    def dot(self, o):
        return matmul(self, o)

    def max(self, axis=None):
        return _reduce_select(self, axis, "max")

    def min(self, axis=None):
        return _reduce_select(self, axis, "min")

    # ---- numpy protocol: implemented operations run in the format, the rest raise
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if kwargs.get("out") is not None:
            raise TypeError("pfloat: out= is not supported")
        if method == "reduce" and ufunc is np.add:
            (x,) = inputs
            return sum(x, axis=kwargs.get("axis", 0), keepdims=kwargs.get("keepdims", False))
        if method != "__call__" or kwargs:
            raise TypeError(f"pfloat: numpy {ufunc.__name__}.{method} is not implemented at p-bit precision")
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
        if name == "negative":
            return _with_sign(inputs[0], "negative")
        if name == "positive":
            return inputs[0].copy()
        if name in ("absolute", "fabs"):
            return absolute(inputs[0])
        if name in ("maximum", "fmax"):
            return maximum(*inputs)
        if name in ("minimum", "fmin"):
            return minimum(*inputs)
        if name in _CMP:
            return _compare(name, *inputs)
        if name in ("isnan", "isinf", "isfinite", "signbit"):
            return _flags(inputs[0], name)
        if name in ("floor", "ceil", "trunc", "rint", "copysign") and not any(
                isinstance(a, PArray) and a.fmt.extended for a in inputs):
            fmt = next(a.fmt for a in inputs if isinstance(a, PArray))
            vals = [a._v if isinstance(a, PArray) else array(a, fmt)._v for a in inputs]
            return PArray._wrap(ufunc(*vals), fmt)
        raise TypeError(f"pfloat: numpy ufunc '{name}' is not implemented at p-bit precision. Convert "
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
            np.max: lambda a, axis=None, **k: a.max(axis), np.min: lambda a, axis=None, **k: a.min(axis),
            np.array_equal: lambda a, b, *r, **k: bool(np.all(_compare("equal", a, b))) if np.shape(a) == np.shape(b) else False,
        }
        if func in handled:
            return handled[func](*args, **kwargs)
        raise TypeError(f"pfloat: numpy.{func.__name__} is not implemented at p-bit precision. Convert "
                        "explicitly with .to_numpy() if you intend to leave the format.")


def _mpf_exact(v: Fraction):
    """An mpmath number equal to the dyadic rational v."""
    import mpmath
    num, den = v.numerator, v.denominator
    with mpmath.workprec(max(53, abs(num).bit_length())):
        return mpmath.ldexp(mpmath.mpf(num), -(den.bit_length() - 1))
