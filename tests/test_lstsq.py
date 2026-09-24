"""The DGELSS port against netlib reference LAPACK SGELSS/DGELSS (compiled from source), and its
behaviour in other formats."""
import numpy as np
import pytest

import pbit
from pbit.testing import reference_lapack

needs_reference = pytest.mark.skipif(not reference_lapack.available(),
                                     reason="needs gfortran and third_party/lapack-3.12.1")

# (m, n, nrhs): every DGELSS path. M >= N with and without the QR step, M < N through the LQ
# step and through direct bidiagonalization (including very wide, where the workspace rule
# switches), several right-hand sides.
SHAPES = [(40, 7, 1), (12, 10, 1), (9, 9, 1), (1, 1, 1), (6, 20, 1), (10, 13, 1), (5, 200, 1),
          (30, 8, 3), (5, 30, 3), (10, 14, 3), (60, 25, 2)]


def _matrices(rng, m, n):
    yield "random", rng.standard_normal((m, n))
    k = max(1, min(m, n) // 2)
    yield "rank-deficient", rng.standard_normal((m, k)) @ rng.standard_normal((k, n))
    yield "graded", rng.standard_normal((m, n)) * np.logspace(0, -10, n)[None, :]


@needs_reference
@pytest.mark.parametrize("fmt,dtype", [(pbit.FP32, np.float32), (pbit.FP64, np.float64)], ids=["fp32", "fp64"])
@pytest.mark.parametrize("m,n,nrhs", SHAPES)
def test_bit_identical_to_reference(fmt, dtype, m, n, nrhs):
    rng = np.random.default_rng(m * 100 + n * 10 + nrhs)
    tiny, huge = (1e-300, 1e300) if dtype == np.float64 else (1e-36, 1e36)
    for kind, a0 in _matrices(rng, m, n):
        for scale in (1.0, tiny, huge):  # tiny and huge trigger DLASCL
            a = pbit.array(a0 * scale, fmt)
            b = pbit.array(rng.standard_normal((m, nrhs)) if nrhs > 1 else rng.standard_normal(m), fmt)
            for rcond in (fmt.eps * max(m, n), -1.0, 1e-3):
                rc = float(pbit.array(rcond, fmt))
                ref = reference_lapack.gelss(a.to_numpy(), b.to_numpy(), rc, dtype)
                assert ref["info"] == 0
                for backend in ("emulator", "auto"):
                    (x,), (rank,), s = pbit.linalg._solve(a, b, [rc], backend)
                    msg = f"{kind} scale={scale} rcond={rcond} {backend}"
                    np.testing.assert_array_equal(x.to_numpy(), ref["x"], err_msg=msg)
                    np.testing.assert_array_equal(s.to_numpy(), ref["sigma"], err_msg=msg)
                    assert rank == ref["rank"], msg


@needs_reference
@pytest.mark.parametrize("fmt,dtype", [(pbit.FP32, np.float32), (pbit.FP64, np.float64)], ids=["fp32", "fp64"])
def test_zero_matrix_matches_reference(fmt, dtype):
    a, b = pbit.zeros((7, 4), fmt), pbit.array(np.arange(7.0), fmt)
    ref = reference_lapack.gelss(a.to_numpy(), b.to_numpy(), fmt.eps, dtype)
    x, _, rank, s = pbit.lstsq(a, b, rcond=fmt.eps)
    np.testing.assert_array_equal(x.to_numpy(), ref["x"])
    assert rank == ref["rank"] == 0


@pytest.mark.parametrize("threads", [1, 3, 8])
def test_thread_count_does_not_change_bits(threads):
    rng = np.random.default_rng(1)
    fmt = pbit.Format(29)
    a, b = pbit.array(rng.standard_normal((300, 120)), fmt), pbit.array(rng.standard_normal((300, 2)), fmt)
    old = pbit.get_num_threads()
    try:
        pbit.set_num_threads(1)
        x1, _, _, s1 = pbit.lstsq(a, b)
        pbit.set_num_threads(threads)
        xt, _, _, st = pbit.lstsq(a, b)
    finally:
        pbit.set_num_threads(old)
    np.testing.assert_array_equal(x1.to_numpy(), xt.to_numpy())
    np.testing.assert_array_equal(s1.to_numpy(), st.to_numpy())


def test_numpy_compatible_signature_and_fp64_agreement():
    rng = np.random.default_rng(2)
    a0, b0 = rng.standard_normal((50, 6)), rng.standard_normal(50)
    x, res, rank, s = pbit.lstsq(a0, b0, fmt=pbit.FP64)
    xn, resn, rankn, sn = np.linalg.lstsq(a0, b0, rcond=None)
    assert x.shape == xn.shape and res.shape == resn.shape and rank == rankn and s.shape == sn.shape
    np.testing.assert_allclose(x.to_numpy(), xn, rtol=1e-12)
    np.testing.assert_allclose(res.to_numpy(), resn, rtol=1e-10)
    np.testing.assert_allclose(s.to_numpy(), sn, rtol=1e-13)
    # matrix right-hand side
    x2, res2, _, _ = pbit.lstsq(a0, np.stack([b0, 2 * b0], axis=1), fmt=pbit.FP64)
    assert x2.shape == (6, 2) and res2.shape == (2,)
    # np.linalg.lstsq dispatches to pbit for PArrays
    xa, *_ = np.linalg.lstsq(pbit.array(a0, 30), pbit.array(b0, 30))
    assert isinstance(xa, pbit.PArray) and xa.fmt == pbit.Format(30)


def test_results_are_format_values_and_residuals_are_computed_in_format():
    rng = np.random.default_rng(3)
    fmt = pbit.Format(13)
    a, b = pbit.array(rng.standard_normal((40, 5)), fmt), pbit.array(rng.standard_normal(40), fmt)
    x, res, rank, s = pbit.lstsq(a, b)
    for arr in (x, res, s):
        np.testing.assert_array_equal(pbit.round_to(arr.to_numpy(), fmt), arr.to_numpy())
    r = a @ x - b
    np.testing.assert_array_equal(res.to_numpy(), pbit.sum(r * r, axis=0).reshape(1).to_numpy())


def test_cutoffs_equal_separate_calls():
    rng = np.random.default_rng(4)
    fmt = pbit.Format(20)
    a = pbit.array(rng.standard_normal((60, 20)) * np.logspace(0, -6, 20), fmt)
    b = pbit.array(rng.standard_normal(60), fmt)
    rconds = [1e-6, 1e-3, 1e-1]
    xs, ranks, s = pbit.lstsq_cutoffs(a, b, rconds)
    for rc, x, r in zip(rconds, xs, ranks):
        xi, _, ri, si = pbit.lstsq(a, b, rcond=rc)
        np.testing.assert_array_equal(x.to_numpy(), xi.to_numpy())
        assert r == ri
    assert ranks[0] >= ranks[1] >= ranks[2]


@pytest.mark.parametrize("fmt", [pbit.FP16, pbit.BF16, pbit.Format(12)], ids=repr)
def test_narrow_formats_solve_well_conditioned_systems(fmt):
    """In a narrow format the solution of a well-conditioned system is accurate to a small
    multiple of the unit roundoff (and DLASCL scaling is exercised for FP16's small range)."""
    rng = np.random.default_rng(5)
    a0 = rng.standard_normal((30, 4))
    x_true = rng.standard_normal(4)
    a = pbit.array(a0, fmt)
    b = a @ pbit.array(x_true, fmt)
    x, *_ = pbit.lstsq(a, b)
    err = np.max(np.abs(x.to_numpy() - x_true)) / np.max(np.abs(x_true))
    assert err < 200 * fmt.unit_roundoff
