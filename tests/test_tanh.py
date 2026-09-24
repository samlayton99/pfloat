"""tanh in the format: within 3 ulp of the true value for p = 8..53, and identical between the
emulator and native binary32/binary64."""
import numpy as np
import pytest

import pbit
from pbit import _lib


def _tanh(fmt, z, kind="emul"):
    z = np.ascontiguousarray(z, dtype=np.float64)
    out = np.empty_like(z)
    _lib.library(kind).pb_unary(_lib.ptr(fmt.constants()), 1, z.size, _lib.ptr(z), _lib.ptr(out))
    return out


def _max_ulp_error(fmt, z, t):
    import gmpy2
    worst = 0.0
    with gmpy2.context(precision=300):
        for zi, ti in zip(z, t):
            true = gmpy2.tanh(gmpy2.mpfr(float(zi)))
            if true == 0:
                assert ti == 0
                continue
            e = max(int(gmpy2.floor(gmpy2.log2(abs(true)))), fmt.emin)
            ulp = gmpy2.exp2(e - fmt.p + 1)
            worst = max(worst, float(abs(gmpy2.mpfr(float(ti)) - true) / ulp))
    return worst


@pytest.mark.parametrize("p", range(8, 54))
def test_within_three_ulp(p):
    fmt = pbit.Format(p)
    rng = np.random.default_rng(1000 + p)
    z = np.unique(pbit.round_to(np.concatenate([rng.uniform(-12, 12, 500), np.exp2(rng.uniform(-30, 3, 200)),
                                                [0.0, 1e-300]]), fmt))
    assert _max_ulp_error(fmt, z, _tanh(fmt, z)) <= 3.0


@pytest.mark.parametrize("fmt", [pbit.FP16, pbit.BF16, pbit.FP32], ids=repr)
def test_presets(fmt):
    rng = np.random.default_rng(fmt.p)
    z = np.unique(pbit.round_to(rng.uniform(-8, 8, 800), fmt))
    assert _max_ulp_error(fmt, z, _tanh(fmt, z)) <= 3.0


@pytest.mark.parametrize("fmt,kind", [(pbit.FP32, "f32"), (pbit.FP64, "f64")])
def test_native_equals_emulator(fmt, kind):
    z = pbit.round_to(np.random.default_rng(0).uniform(-20, 20, 20000), fmt)
    np.testing.assert_array_equal(_tanh(fmt, z, kind), _tanh(fmt, z))


def test_specials():
    fmt = pbit.Format(20)
    out = _tanh(fmt, np.array([np.inf, -np.inf, np.nan, -0.0]))
    assert out[0] == 1 and out[1] == -1 and np.isnan(out[2]) and out[3] == 0
