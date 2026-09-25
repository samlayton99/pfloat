"""Every +, -, *, /, sqrt is the correctly rounded result in the format.

Oracles: MPFR (gmpy2) with IEEE subnormals for every p and the presets; numpy's float32 / float64
/ float16 hardware arithmetic; the native builds.
"""
import numpy as np
import pytest

import pfloat
from conftest import emulated, mpfr_op, random_values

OPS = ("add", "sub", "mul", "div", "sqrt")


def _check(fmt, a, b):
    for name in OPS:
        x = np.abs(a) if name == "sqrt" else a
        np.testing.assert_array_equal(emulated(fmt, name, x, b), mpfr_op(fmt, name, x, b), err_msg=f"{fmt} {name}")


def _midpoint_cases(p):
    """Operand pairs whose exact results sit on or next to rounding midpoints."""
    ulp = 2.0 ** (1 - p)
    a, b = [], []
    for k in range(4):
        for j in (1, 2, 3):
            for s in range(4):
                a += [1 + k * ulp, 1 + k * ulp]
                b += [j * ulp / 2 ** (s + 1), -j * ulp / 2 ** (s + 1)]
    for i in range(1, 6):
        for j in range(1, 6):
            a += [1 + i * ulp, 1 - i * ulp / 2]
            b += [1 + j * ulp, 1 + j * ulp]
    return np.array(a), np.array(b)


@pytest.mark.parametrize("p", range(2, 54))
def test_default_range_every_p(p):
    fmt = pfloat.Format(p)
    rng = np.random.default_rng(p)
    sa, sb = _midpoint_cases(p)
    a = np.concatenate([random_values(rng, fmt, 1500, -8, 8), pfloat.round_to(sa, fmt),
                        random_values(rng, fmt, 300, fmt.emin - p + 1, fmt.emin + 3),   # subnormal range
                        random_values(rng, fmt, 200, fmt.emax - 3, fmt.emax)])           # near overflow
    b = np.concatenate([random_values(rng, fmt, 1500, -8, 8), pfloat.round_to(sb, fmt),
                        random_values(rng, fmt, 300, -3, 3), random_values(rng, fmt, 200, -2, 4)])
    _check(fmt, a, b)


@pytest.mark.parametrize("fmt", [pfloat.FP32, pfloat.FP16, pfloat.BF16, pfloat.TF32, pfloat.FP8_E5M2],
                         ids=lambda f: repr(f))
def test_presets_with_subnormals_and_overflow(fmt):
    rng = np.random.default_rng(fmt.p * 7 + fmt.emax)
    a = np.concatenate([random_values(rng, fmt, 3000, fmt.emin - fmt.p + 1, fmt.emax),
                        random_values(rng, fmt, 1000, -3, 3)])
    b = np.concatenate([random_values(rng, fmt, 3000, -6, 6),
                        random_values(rng, fmt, 1000, fmt.emin - fmt.p + 1, fmt.emax)])
    _check(fmt, a, b)


@pytest.mark.parametrize("p", [11, 24, 40, 52, 53])
def test_tiny_results(p):
    """Products, quotients and roots near 2^-1000, where an unscaled FMA residual could underflow."""
    fmt = pfloat.Format(p)
    rng = np.random.default_rng(500 + p)
    a = random_values(rng, fmt, 3000, -560, -420)
    b = random_values(rng, fmt, 3000, -560, -420)
    for name in ("mul", "div", "sqrt"):
        x = np.abs(a) if name == "sqrt" else a
        y = pfloat.round_to(np.ldexp(b, 480), fmt) if name == "div" else b
        np.testing.assert_array_equal(emulated(fmt, name, x, y), mpfr_op(fmt, name, x, y), err_msg=name)


def test_double_rounding_trap():
    # The exact product lies just below a p = 52 midpoint; rounding through binary64 first would
    # land on the midpoint and round up.
    fmt = pfloat.Format(52)
    a, b = np.array([1 + 2.0 ** -26]), np.array([1 + 2.0 ** -26 - 2.0 ** -51])
    out = emulated(fmt, "mul", a, b)
    np.testing.assert_array_equal(out, mpfr_op(fmt, "mul", a, b))
    assert out[0] != pfloat.round_to(a * b, fmt)[0]


def test_specials_follow_ieee():
    fmt = pfloat.Format(12)
    a = np.array([1.0, -1.0, 0.0, np.inf, np.inf, 0.0, -0.0])
    b = np.array([0.0, 0.0, 0.0, 1.0, -np.inf, 5.0, 5.0])
    div = emulated(fmt, "div", a, b)
    assert div[0] == np.inf and div[1] == -np.inf and np.isnan(div[2]) and div[3] == np.inf
    assert np.isnan(emulated(fmt, "add", a, b)[4])
    assert np.signbit(emulated(fmt, "mul", a, b)[6])
    assert np.isnan(emulated(fmt, "sqrt", np.array([-1.0]))[0])


@pytest.mark.parametrize("fmt,dtype", [(pfloat.FP32, np.float32), (pfloat.FP64, np.float64), (pfloat.FP16, np.float16)],
                         ids=["fp32", "fp64", "fp16"])
def test_matches_hardware(fmt, dtype):
    """numpy float32/float64 arithmetic is IEEE hardware; numpy float16 rounds a float32 result,
    which is exact for p = 11. Subnormals and overflow included."""
    rng = np.random.default_rng(3)
    lo = fmt.emin - fmt.p + 1
    a = np.concatenate([rng.standard_normal(20000), np.ldexp(rng.standard_normal(3000), rng.integers(lo, 0, 3000)),
                        np.ldexp(rng.standard_normal(1000), rng.integers(0, fmt.emax, 1000))]).astype(dtype)
    b = np.concatenate([rng.standard_normal(20000), rng.standard_normal(3000),
                        np.ldexp(rng.standard_normal(1000), rng.integers(0, 4, 1000))]).astype(dtype)
    with np.errstate(all="ignore"):
        hw = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b, "sqrt": np.sqrt(np.abs(a))}
    for name, ref in hw.items():
        x = (np.abs(a) if name == "sqrt" else a).astype(np.float64)
        got = emulated(fmt, name, x, b.astype(np.float64))
        np.testing.assert_array_equal(got, ref.astype(np.float64), err_msg=name)
        if fmt in (pfloat.FP32, pfloat.FP64):
            kind = "f32" if fmt == pfloat.FP32 else "f64"
            np.testing.assert_array_equal(emulated(fmt, name, x, b.astype(np.float64), kind=kind), got, err_msg=name)


def test_round_to_matches_mpfr_and_rational_rounding():
    import gmpy2
    from conftest import mpfr_context
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.standard_normal(3000) * np.exp2(rng.integers(-40, 40, 3000)),
                        np.ldexp(1 + np.arange(64) / 64.0, -1000)])
    for fmt in (pfloat.Format(7), pfloat.Format(23), pfloat.FP16, pfloat.BF16, pfloat.Format(52)):
        with mpfr_context(fmt):
            want = np.array([float(gmpy2.mpfr(float(v))) for v in x])
        np.testing.assert_array_equal(pfloat.round_to(x, fmt), want)
        np.testing.assert_array_equal([pfloat.round_rational(v, fmt) for v in x[:300]], want[:300])
