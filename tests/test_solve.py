"""solve / lu_factor / lu_solve (DGESV), cholesky / cho_factor / cho_solve (DPOTRF, DPOTRS) and
norm (DNRM2): bit-identical to netlib reference LAPACK in binary32/binary64 on both backends,
and accurate to the format's unit roundoff at other p."""
from fractions import Fraction

import numpy as np
import pytest

import pfloat
from pfloat.linalg import LinAlgError, cho_factor, cho_solve, lu_factor, lu_solve

from pfloat.testing import reference_lapack as ref

needs_reference = pytest.mark.skipif(not ref.available(), reason="reference LAPACK needs gfortran")

BACKENDS = ["auto", "emulator"]
IEEE = [(np.float64, pfloat.FP64), (np.float32, pfloat.FP32)]


def rounded(x, dt):
    return np.asarray(x).astype(dt).astype(np.float64)


def spd(rng, n, dt):
    g = rng.standard_normal((n, n))
    return rounded(g @ g.T + n * np.eye(n), dt)


@needs_reference
@pytest.mark.parametrize("dt,fmt", IEEE, ids=["fp64", "fp32"])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [1, 2, 5, 16, 33, 90])
def test_gesv_bit_identical(dt, fmt, backend, n):
    rng = np.random.default_rng(n)
    A = rounded(rng.standard_normal((n, n)) * np.logspace(0, -4, n), dt)
    for b in (rounded(rng.standard_normal(n), dt), rounded(rng.standard_normal((n, 3)), dt)):
        want = ref.gesv(A, b, dt)
        lu, piv = lu_factor(pfloat.array(A, fmt), backend=backend)
        np.testing.assert_array_equal(lu.to_numpy(), want["lu"])
        np.testing.assert_array_equal(piv + 1, want["ipiv"])
        x = pfloat.solve(pfloat.array(A, fmt), b, backend=backend)
        assert x.shape == b.shape
        np.testing.assert_array_equal(x.to_numpy(), want["x"])
        np.testing.assert_array_equal(lu_solve((lu, piv), b, backend=backend).to_numpy(), want["x"])


@needs_reference
@pytest.mark.parametrize("dt,fmt,tiny", [(np.float64, pfloat.FP64, 1e-300), (np.float32, pfloat.FP32, 1e-36)],
                         ids=["fp64", "fp32"])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", [(9, 4), (4, 9), (12, 12)])
def test_getrf_rectangular_and_tiny_pivots(dt, fmt, tiny, backend, shape):
    """Rectangular DGETRF, and pivots below DLAMCH('S'), where DGETRF2 divides instead of scaling
    by the reciprocal (entries down into the subnormal range)."""
    rng = np.random.default_rng(sum(shape))
    for scale in (1.0, tiny):
        A = rounded(rng.standard_normal(shape) * scale * np.logspace(0, -3, shape[1]), dt)
        want = ref.getrf(A, dt)
        assert want["info"] == 0
        lu, piv = lu_factor(pfloat.array(A, fmt), backend=backend)
        np.testing.assert_array_equal(lu.to_numpy(), want["lu"])
        np.testing.assert_array_equal(piv + 1, want["ipiv"])


@needs_reference
@pytest.mark.parametrize("dt,fmt", IEEE, ids=["fp64", "fp32"])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [1, 2, 7, 32, 81])
@pytest.mark.parametrize("lower", [True, False])
def test_potrf_potrs_bit_identical(dt, fmt, backend, n, lower):
    rng = np.random.default_rng(10 * n + lower)
    S = spd(rng, n, dt)
    S_junk = S.copy()   # DPOTRF reads one triangle only; garbage in the other must not matter
    S_junk[np.triu_indices(n, 1) if lower else np.tril_indices(n, -1)] = 7.0
    uplo = "L" if lower else "U"
    want = ref.potrf(S_junk, uplo, dt)
    c, low = cho_factor(pfloat.array(S_junk, fmt), lower=lower, backend=backend)
    assert low is lower
    np.testing.assert_array_equal(c.to_numpy(), want["c"])
    b = rounded(rng.standard_normal((n, 2)), dt)
    np.testing.assert_array_equal(cho_solve((c, low), b, backend=backend).to_numpy(),
                                  ref.potrs(want["c"], b, uplo, dt))
    factor = pfloat.cholesky(pfloat.array(S_junk, fmt), upper=not lower, backend=backend).to_numpy()
    np.testing.assert_array_equal(factor, np.tril(want["c"]) if lower else np.triu(want["c"]))


@needs_reference
@pytest.mark.parametrize("dt,fmt", IEEE, ids=["fp64", "fp32"])
def test_singular_and_indefinite_raise(dt, fmt):
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    assert ref.getrf(A, dt)["info"] > 0
    with pytest.raises(LinAlgError, match="singular"):
        pfloat.solve(pfloat.array(A, fmt), [1, 2, 3])
    for S, order in (([[4.0, 2.0], [2.0, -1.0]], 2), ([[-1.0, 0.0], [0.0, 1.0]], 1),
                     ([[np.nan, 0.0], [0.0, 1.0]], 1)):
        assert ref.potrf(np.array(S), "L", dt)["info"] == order
        with pytest.raises(LinAlgError, match=f"order {order}"):
            pfloat.cholesky(pfloat.array(S, fmt))


@needs_reference
@pytest.mark.parametrize("dt,fmt", IEEE, ids=["fp64", "fp32"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_nrm2_bit_identical_across_ranges(dt, fmt, backend):
    """DNRM2's three accumulators: tiny, mid and huge magnitudes, and mixtures of them."""
    rng = np.random.default_rng(3)
    big = float(np.finfo(dt).max) / 4
    small = float(np.finfo(dt).tiny) * 64
    for trial in range(40):
        n = int(rng.integers(1, 60))
        scales = rng.choice([small, 1.0, big, 1e-3, 1e3], size=n) if trial % 2 else np.full(n, [small, 1.0, big][trial % 3])
        v = rounded(rng.standard_normal(n) * scales, dt)
        assert float(pfloat.norm(pfloat.array(v, fmt), backend=backend)) == ref.nrm2(v, dt), trial


def test_norm_axes_and_keepdims():
    x = pfloat.array(np.arange(12.0).reshape(3, 4), pfloat.FP64)
    np.testing.assert_allclose(pfloat.norm(x).to_numpy(), np.linalg.norm(np.arange(12.0)), rtol=1e-15)
    np.testing.assert_allclose(pfloat.norm(x, axis=0).to_numpy(), np.linalg.norm(np.arange(12.0).reshape(3, 4), axis=0),
                               rtol=1e-15)
    assert pfloat.norm(x, axis=1, keepdims=True).shape == (3, 1)
    assert pfloat.norm(x, keepdims=True).shape == (1, 1)
    assert float(pfloat.norm([3.0, 4.0], fmt=11)) == 5.0


@pytest.mark.parametrize("p", [8, 11, 16, 24, 30, 40, 53])
def test_solve_backward_error_at_p(p):
    """At precision p the computed LU solution has a small normwise backward error: the residual
    b - A x, computed exactly, is a modest multiple of n u |A| |x| (u = 2^-p), and likewise for
    Cholesky."""
    fmt = pfloat.Format(p)
    n = 24
    rng = np.random.default_rng(p)
    A = pfloat.array(rng.standard_normal((n, n)), fmt)
    b = pfloat.array(rng.standard_normal(n), fmt)
    S = pfloat.array(spd(rng, n, np.float64), fmt)
    u = float(fmt.unit_roundoff)
    for M, x in ((A, pfloat.solve(A, b)), (S, pfloat.linalg.cho_solve(cho_factor(S), b))):
        Af, xf, bf = M.to_fractions(), x.to_fractions(), b.to_fractions()
        for i in range(n):
            r = bf[i] - sum(Af[i, j] * xf[j] for j in range(n))
            scale = sum(abs(Af[i, j] * xf[j]) for j in range(n))
            assert abs(float(r)) <= 4 * n * u * float(scale)


def test_threads_do_not_change_results():
    rng = np.random.default_rng(0)
    n = 160
    A = pfloat.array(rng.standard_normal((n, n)), pfloat.Format(30))
    S = pfloat.array(spd(rng, n, np.float64), pfloat.Format(30))
    b = pfloat.array(rng.standard_normal((n, 5)), pfloat.Format(30))
    old = pfloat.get_num_threads()
    try:
        out = []
        for t in (1, 6):
            pfloat.set_num_threads(t)
            out.append((pfloat.solve(A, b).to_numpy(), pfloat.cholesky(S).to_numpy(), cho_solve(cho_factor(S), b).to_numpy()))
    finally:
        pfloat.set_num_threads(old)
    for u, v in zip(*out):
        np.testing.assert_array_equal(u, v)


def test_api_shapes_and_errors():
    fmt = pfloat.FP64
    with pytest.raises(ValueError, match="square"):
        pfloat.solve(pfloat.array(np.ones((3, 2)), fmt), [1, 2, 3])
    with pytest.raises(ValueError, match="square"):
        pfloat.cholesky(pfloat.array(np.ones((3, 2)), fmt))
    with pytest.raises(TypeError, match="mixed formats"):
        cho_solve(cho_factor(pfloat.array(np.eye(2), fmt)), pfloat.array([1, 2], pfloat.FP32))
    with pytest.raises(ValueError, match="backend"):
        pfloat.solve(pfloat.array(np.eye(2), fmt), [1, 2], backend="gpu")
    lu, piv = lu_factor(pfloat.array(np.zeros((0, 0)), fmt))
    assert lu.shape == (0, 0) and piv.shape == (0,)
    x = pfloat.solve(np.eye(3), [1, 2, 3], fmt=16)
    assert x.fmt == pfloat.Format(16) and [float(v) for v in x] == [1.0, 2.0, 3.0]
    assert pfloat.cholesky([[4.0]], fmt=16).to_fractions()[0, 0] == Fraction(2)
