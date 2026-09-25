"""Formats with p > 53 (the MPFR-backed build): arithmetic, tanh, lstsq, the array API, and a
cross-check of the MPFR build against the binary64 emulator at p <= 53 through DGELSS."""
from fractions import Fraction

import numpy as np
import pytest

import pfloat
from pfloat import _lib, _mp
from pfloat._formats import exact_constants

try:
    _mp.library(1)
except RuntimeError as err:  # no MPFR or no C++ compiler
    pytest.skip(f"extended precision unavailable: {err}", allow_module_level=True)

gmpy2 = pytest.importorskip("gmpy2")


def ctx(fmt):
    return gmpy2.context(precision=fmt.p, emin=fmt.emin - fmt.p + 2, emax=fmt.emax + 1, subnormalize=True,
                         round=gmpy2.RoundToNearest)


def mp(v):
    return gmpy2.mpfr(gmpy2.mpq(v.numerator, v.denominator)) if isinstance(v, Fraction) else gmpy2.mpfr(v)


def frac(x):
    return Fraction(*x.as_integer_ratio()) if gmpy2.is_finite(x) else float(x)


def random_exact(rng, fmt, n, lo, hi):
    e = rng.integers(lo, hi + 1, n)
    m = [int(rng.integers(1, 2 ** 62)) << max(0, fmt.p - 62) for _ in range(n)]
    m = [v | (1 << (fmt.p - 1)) for v in m]
    return [(-1) ** int(rng.integers(2)) * Fraction(v) * Fraction(2) ** int(k - fmt.p + 1) for v, k in zip(m, e)]


@pytest.mark.parametrize("fmt", [pfloat.Format(54), pfloat.Format(64), pfloat.FP128, pfloat.Format(200),
                                 pfloat.FP256, pfloat.Format(1000), pfloat.Format(64, emax=40)], ids=repr)
def test_arithmetic_matches_mpfr(fmt):
    rng = np.random.default_rng(fmt.p)
    lo = max(fmt.emin - fmt.p + 1, -60)
    a_vals = random_exact(rng, fmt, 300, lo, min(fmt.emax, 60))
    b_vals = random_exact(rng, fmt, 300, -20, min(fmt.emax, 20))
    a, b = pfloat.array(a_vals, fmt), pfloat.array(b_vals, fmt)
    with ctx(fmt):
        ga = [mp(v) for v in a.to_fractions().ravel()]
        gb = [mp(v) for v in b.to_fractions().ravel()]
        want = {"add": [x + y for x, y in zip(ga, gb)], "sub": [x - y for x, y in zip(ga, gb)],
                "mul": [x * y for x, y in zip(ga, gb)], "div": [x / y for x, y in zip(ga, gb)],
                "sqrt": [gmpy2.sqrt(abs(x)) for x in ga]}
    got = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b, "sqrt": pfloat.sqrt(abs(a))}
    for name in want:
        assert list(got[name].to_fractions().ravel()) == [frac(w) for w in want[name]], name


def test_tiny_format_subnormals_and_overflow():
    fmt = pfloat.Format(70, emax=20)
    a = pfloat.array([Fraction(3, 2 ** 25), 2.0 ** 19 * 1.5, -(2.0 ** -40)], fmt)
    b = pfloat.array([Fraction(1, 2 ** 70), 4.0, 2.0 ** -45], fmt)
    with ctx(fmt):
        want = [frac(mp(x) * mp(y)) for x, y in zip(a.to_fractions(), b.to_fractions())]
    assert list((a * b).to_fractions()) == want
    assert (a * b).to_fractions()[1] == float("inf")


def _cross_solve(fmt, A, b, rcond):
    """DGELSS through the MPFR build for a p <= 53 format (values as records)."""
    L = 1
    lib = _mp.library(L)
    m, n = A.shape
    enc = lambda xs: _mp.encode([Fraction(v) if np.isfinite(v) and v != 0 else float(v) for v in xs], fmt.p, L)  # noqa: E731
    Arec = enc(np.ravel(A, order="F"))
    Brec = enc(np.concatenate([b, np.zeros(max(m, n) - m)]))
    vals, coeffs = exact_constants(fmt.p, fmt.emin, fmt.emax)
    cv = _mp.encode([Fraction(v) for v in [fmt.p, fmt.emin, fmt.emax, *vals, len(coeffs), *coeffs]], fmt.p, L)
    X, S = np.zeros(n, dtype=_mp.dtype(L)), np.zeros(min(m, n), dtype=_mp.dtype(L))
    ranks = np.zeros(1, dtype=np.intc)
    info = lib.pb_gelss(_lib.ptr(fmt.constants()), _mp.vptr(cv), m, n, 1, _mp.vptr(Arec), _mp.vptr(Brec), 1,
                        _mp.vptr(enc([rcond])), _mp.vptr(X), _mp.vptr(S), _lib.iptr(ranks))
    assert info == 0
    dec = lambda r: np.array([float(v) for v in _mp.decode(r, fmt.p)])  # noqa: E731
    return dec(X), dec(S), int(ranks[0])


@pytest.mark.parametrize("p", [11, 24, 40, 53])
@pytest.mark.parametrize("m,n", [(60, 12), (12, 10), (8, 20), (5, 120)])
def test_mpfr_build_equals_emulator(p, m, n):
    """Two independent implementations of the p-bit rounding (binary64 with error-free transforms,
    and MPFR) running the same DGELSS code must agree bit for bit."""
    fmt = pfloat.Format(p)
    rng = np.random.default_rng(p + m + n)
    A = pfloat.array(rng.standard_normal((m, n)) * np.logspace(0, -5, n), fmt)
    b = pfloat.array(rng.standard_normal(m), fmt)
    (x,), (rank,), s = pfloat.linalg._solve(A, b, [float(fmt.eps)], "emulator")
    xm, sm, rm = _cross_solve(fmt, A.to_numpy(), b.to_numpy(), float(fmt.eps))
    np.testing.assert_array_equal(xm, x.to_numpy())
    np.testing.assert_array_equal(sm, s.to_numpy())
    assert rm == rank


@pytest.mark.parametrize("fmt", [pfloat.Format(64), pfloat.FP128, pfloat.Format(300)], ids=repr)
def test_tanh_within_three_ulp(fmt):
    rng = np.random.default_rng(fmt.p)
    z = pfloat.array(rng.uniform(-10, 10, 200), fmt)
    t = pfloat.tanh(z).to_fractions()
    with gmpy2.context(precision=fmt.p + 200):
        for zi, ti in zip(z.to_fractions(), t):
            true = gmpy2.tanh(mp(zi))
            ulp = gmpy2.exp2(gmpy2.floor(gmpy2.log2(abs(true))) - fmt.p + 1)
            assert abs(mp(ti) - true) / ulp <= 3


def test_lstsq_binary128_accuracy_and_threads():
    rng = np.random.default_rng(1)
    A0, x0 = rng.standard_normal((50, 7)), rng.standard_normal(7)
    A = pfloat.array(A0, pfloat.FP128)
    b = A @ pfloat.array(x0, pfloat.FP128)
    old = pfloat.get_num_threads()
    try:
        pfloat.set_num_threads(1)
        x1, _, rank, s = pfloat.lstsq(A, b)
        pfloat.set_num_threads(4)
        x4, *_ = pfloat.lstsq(A, b)
    finally:
        pfloat.set_num_threads(old)
    assert rank == 7
    assert list(x1.to_fractions()) == list(x4.to_fractions())
    err = max(abs(v - Fraction(t)) for v, t in zip(x1.to_fractions(), x0))
    assert err < 1000 * pfloat.FP128.unit_roundoff


def test_array_api_extended():
    import mpmath
    fmt = pfloat.FP128
    third = pfloat.array(Fraction(1, 3), fmt)
    with ctx(fmt):
        assert third.to_fractions()[()] == frac(gmpy2.mpfr(1) / 3)
    with mpmath.workprec(300):
        assert pfloat.array(mpmath.mpf(1) / 3, fmt).to_fractions()[()] == third.to_fractions()[()]
    x = pfloat.array([1.5, -2.0, 0.25], fmt)
    np.testing.assert_array_equal(x.to_numpy(), [1.5, -2.0, 0.25])       # exact: allowed
    with pytest.raises(ValueError):
        (x / 3).to_numpy()                                                # not exact in binary64
    assert abs((x / 3).to_numpy(rounding=True)[0] - 0.5) < 1e-16
    assert list((-x).to_numpy()) == [-1.5, 2.0, -0.25] and list(abs(x).to_numpy()) == [1.5, 2.0, 0.25]
    assert list(x > 0) == [True, False, True] and float(x.max()) == 1.5 and float(x.min()) == -2.0
    assert x.sum().to_fractions()[()] == Fraction(-1, 4)
    assert "fp128" in repr(x / 3) and "0.5" in repr(x / 3)
    down = (x / 3).astype(pfloat.FP32)
    np.testing.assert_array_equal(down.to_numpy(), np.float32([0.5, -2 / 3, 1 / 12]).astype(np.float64))
    up = pfloat.array([0.1], pfloat.FP32).astype(fmt)
    assert up.to_fractions()[0] == Fraction(float(np.float32(0.1)))
    assert pfloat.array(0.0, fmt).to_fractions()[()] == 0 and np.isnan(pfloat.array(np.nan, fmt).to_numpy())
    assert (x @ x).to_fractions()[()] == Fraction(9, 4) + 4 + Fraction(1, 16)


def _mp_call(fmt, name, ints, mats, outs, iarrays=()):
    """Call an MPFR-build kernel on a p <= 53 format: mats are column-major float arrays (encoded
    as records), outs the output sizes; iarrays int32 arrays passed (and returned) by pointer."""
    L = 1
    lib = _mp.library(L)
    enc = lambda xs: _mp.encode([Fraction(v) if np.isfinite(v) and v != 0 else float(v) for v in xs], fmt.p, L)  # noqa: E731
    vals, coeffs = exact_constants(fmt.p, fmt.emin, fmt.emax)
    cv = _mp.encode([Fraction(v) for v in [fmt.p, fmt.emin, fmt.emax, *vals, len(coeffs), *coeffs]], fmt.p, L)
    ins = [enc(np.ravel(m, order="F")) for m in mats]
    out = [np.zeros(max(k, 1), dtype=_mp.dtype(L)) for k in outs]
    args = [_lib.ptr(fmt.constants()), _mp.vptr(cv), *ints]
    return lib, args, ins, out, lambda r: np.array([float(v) for v in _mp.decode(r, fmt.p)])


@pytest.mark.parametrize("p", [11, 24, 40, 53])
@pytest.mark.parametrize("n", [1, 6, 23])
def test_mpfr_build_equals_emulator_solve_cholesky_norm(p, n):
    """DGETRF/DGETRS, DPOTRF/DPOTRS (both triangles) and DNRM2 through the MPFR build agree bit for
    bit with the binary64-carrier emulator at p <= 53."""
    fmt = pfloat.Format(p)
    rng = np.random.default_rng(p * 100 + n)
    A = pfloat.array(rng.standard_normal((n, n)) * np.logspace(0, -3, n), fmt)
    g = rng.standard_normal((n, n))
    S = pfloat.array(g @ g.T + n * np.eye(n), fmt)
    B = pfloat.array(rng.standard_normal((n, 2)), fmt)
    An, Sn, Bn = A.to_numpy(), S.to_numpy(), B.to_numpy()

    lu, piv = pfloat.linalg.lu_factor(A, backend="emulator")
    lib, args, (a_rec,), (lu_rec,), dec = _mp_call(fmt, "getrf", [n, n], [An], [n * n])
    ipiv = np.zeros(max(n, 1), dtype=np.intc)
    assert lib.pb_getrf(*args, _mp.vptr(a_rec), _mp.vptr(lu_rec), _lib.iptr(ipiv)) == 0
    np.testing.assert_array_equal(dec(lu_rec).reshape((n, n), order="F"), lu.to_numpy())
    np.testing.assert_array_equal(ipiv[:n] - 1, piv)

    x = pfloat.linalg.lu_solve((lu, piv), B, backend="emulator")
    lib, args, (lu_in, b_rec), (x_rec,), dec = _mp_call(fmt, "getrs", [n, 2], [lu.to_numpy(), Bn], [n * 2])
    assert lib.pb_getrs(*args, _mp.vptr(lu_in), _lib.iptr(ipiv), _mp.vptr(b_rec), _mp.vptr(x_rec)) == 0
    np.testing.assert_array_equal(dec(x_rec).reshape((n, 2), order="F"), x.to_numpy())

    for lower in (True, False):
        c, _ = pfloat.linalg.cho_factor(S, lower=lower, backend="emulator")
        lib, args, (s_rec,), (c_rec,), dec = _mp_call(fmt, "potrf", [0 if lower else 1, n], [Sn], [n * n])
        assert lib.pb_potrf(*args, _mp.vptr(s_rec), _mp.vptr(c_rec)) == 0
        np.testing.assert_array_equal(dec(c_rec).reshape((n, n), order="F"), c.to_numpy())
        y = pfloat.linalg.cho_solve((c, lower), B, backend="emulator")
        lib, args, (c_in, b_rec), (y_rec,), dec = _mp_call(fmt, "potrs", [0 if lower else 1, n, 2], [c.to_numpy(), Bn],
                                                           [n * 2])
        assert lib.pb_potrs(*args, _mp.vptr(c_in), _mp.vptr(b_rec), _mp.vptr(y_rec)) == 0
        np.testing.assert_array_equal(dec(y_rec).reshape((n, 2), order="F"), y.to_numpy())

    v = pfloat.array(rng.standard_normal(3 * n) * 10.0 ** rng.integers(-200, 200, 3 * n), fmt)
    lib, args, (v_rec,), (r_rec,), dec = _mp_call(fmt, "nrm2", [1, 3 * n], [v.to_numpy()], [1])
    assert lib.pb_nrm2(*args, _mp.vptr(v_rec), _mp.vptr(r_rec)) == 0
    assert dec(r_rec)[0] == float(pfloat.norm(v, backend="emulator"))


@pytest.mark.parametrize("fmt", [pfloat.FP128, pfloat.Format(300)], ids=repr)
def test_solve_cholesky_norm_extended_accuracy(fmt):
    rng = np.random.default_rng(fmt.p)
    n = 12
    A0, x0 = rng.standard_normal((n, n)), rng.standard_normal(n)
    A = pfloat.array(A0, fmt)
    b = A @ pfloat.array(x0, fmt)
    u = fmt.unit_roundoff
    cond = np.linalg.cond(A0)
    x = pfloat.solve(A, b)
    assert max(abs(v - Fraction(t)) for v, t in zip(x.to_fractions(), x0)) < 100 * n * cond * u
    S = A.T @ A
    L = pfloat.cholesky(S)
    Lf, Sf = L.to_fractions(), S.to_fractions()
    err = max(abs(sum(Lf[i, k] * Lf[j, k] for k in range(n)) - Sf[i, j]) for i in range(n) for j in range(n))
    assert err < 10 * n * u * max(abs(v) for v in Sf.ravel())
    assert all(Lf[i, j] == 0 for i in range(n) for j in range(i + 1, n))
    y = pfloat.linalg.cho_solve(pfloat.linalg.cho_factor(S, lower=True), A.T @ b)
    assert max(abs(v - Fraction(t)) for v, t in zip(y.to_fractions(), x0)) < 100 * n * cond ** 2 * u
    got = pfloat.norm(b).to_fractions()[()]
    with gmpy2.context(precision=fmt.p + 100):
        want = gmpy2.sqrt(mp(sum(v * v for v in b.to_fractions())))
        assert abs(mp(got) - want) <= 4 * mp(u) * want
