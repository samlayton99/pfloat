import numpy as np
import pytest

import pfloat
from pfloat import _lib

gmpy2 = pytest.importorskip("gmpy2", reason="the MPFR oracle needs gmpy2 (pip install pfloat[test])")


def mpfr_context(fmt):
    """A gmpy2 context equal to the format (IEEE-style, gradual underflow)."""
    return gmpy2.context(precision=fmt.p, emin=fmt.emin - fmt.p + 2, emax=fmt.emax + 1,
                         subnormalize=True, round=gmpy2.RoundToNearest)


def random_values(rng, fmt, count, lo, hi):
    """Random values of the format with binary exponents in [lo, hi]."""
    e = rng.integers(lo, hi + 1, count)
    m = rng.integers(2 ** (fmt.p - 1), 2 ** fmt.p, count, dtype=np.int64)
    v = np.ldexp(m.astype(np.float64), e - fmt.p + 1) * rng.choice([-1.0, 1.0], count)
    return pfloat.round_to(v, fmt)


def emulated(fmt, name, a, b=None, kind="emul"):
    """One primitive, elementwise, straight from a kernel build ('emul', 'f32' or 'f64')."""
    lib = _lib.library(kind)
    a = np.ascontiguousarray(a, dtype=np.float64)
    out = np.empty_like(a)
    if name == "sqrt":
        lib.pb_unary(_lib.ptr(fmt.constants()), _lib.ptr(fmt.constants()), 0, a.size, _lib.ptr(a), _lib.ptr(out))
    else:
        b = np.ascontiguousarray(b, dtype=np.float64)
        code = {"add": 0, "sub": 1, "mul": 2, "div": 3}[name]
        lib.pb_binary(_lib.ptr(fmt.constants()), _lib.ptr(fmt.constants()), code, a.size, _lib.ptr(a), _lib.ptr(b), _lib.ptr(out))
    return out


def mpfr_op(fmt, name, a, b):
    ops = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
           "div": lambda x, y: x / y, "sqrt": lambda x, y: gmpy2.sqrt(x)}
    with mpfr_context(fmt):
        return np.array([float(ops[name](gmpy2.mpfr(float(x)), gmpy2.mpfr(float(y)))) for x, y in zip(a, b)])
