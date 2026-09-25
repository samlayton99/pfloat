"""Linear algebra in a p-bit format: reference LAPACK routines ported operation by operation.

lstsq (DGELSS), solve (DGESV), lu_factor / lu_solve (DGETRF / DGETRS), cholesky, cho_factor /
cho_solve (DPOTRF / DPOTRS), norm (DNRM2). In binary32 and binary64 every result is
bit-identical to netlib reference LAPACK 3.12.1 (block size 1); in any other format every
operation of the same algorithm is rounded once into the format.
"""
from __future__ import annotations

import numpy as np

from . import _lib
from ._array import PArray, _cargs, _empty, _kernel, _pair, _ptr, _resolve, array, matmul, sum as _sum
from ._formats import FP32, FP64, Format, as_format, lapack_supported

__all__ = ["lstsq", "lstsq_cutoffs", "solve", "lu_factor", "lu_solve", "cholesky", "cho_factor", "cho_solve",
           "norm", "LinAlgError"]


class LinAlgError(np.linalg.LinAlgError):
    pass


def _prepare(a, b, fmt):
    if not isinstance(a, PArray) and not isinstance(b, PArray):
        fmt = _resolve(fmt)
        a, b = array(a, fmt), array(b, fmt)
    elif fmt is not None:
        fmt = as_format(fmt)
        a, b = array(a, fmt), array(b, fmt)
    else:
        a, b = _pair(a, b)
    if a.ndim != 2:
        raise ValueError("a must be 2-D")
    if b.ndim not in (1, 2) or b.shape[0] != a.shape[0]:
        raise ValueError(f"b must have {a.shape[0]} rows")
    return a, b


def _check_lapack(fmt: Format, what: str) -> None:
    if not lapack_supported(fmt):
        raise ValueError(f"{fmt!r}: the exponent range is too narrow for LAPACK's safe-scaling constants "
                         f"(DLAMCH, DNRM2 thresholds); widen emax to use {what} in this precision")


def _library(fmt: Format, backend: str):
    if backend not in ("auto", "emulator"):
        raise ValueError(f"backend must be 'auto' or 'emulator', got {backend!r}")
    if backend == "emulator" and fmt in (FP32, FP64):
        return _lib.library("emul")
    return _kernel(fmt)


def _colmajor(x: PArray) -> np.ndarray:
    """The values of a 2-D array in column-major order, as a contiguous 1-D buffer."""
    return np.array(np.asfortranarray(x._v).ravel(order="F"), order="C")


def _from_colmajor(buf: np.ndarray, shape, fmt: Format) -> PArray:
    return PArray._wrap(np.ascontiguousarray(buf.reshape(shape, order="F")), fmt)


def _solve(a: PArray, b: PArray, rconds, backend: str):
    """DGELSS on PArrays; rconds are format values (floats for p <= 53, or PArrays)."""
    fmt: Format = a.fmt
    _check_lapack(fmt, "lstsq")
    m, n = a.shape
    nrhs = 1 if b.ndim == 1 else b.shape[1]
    ldb = max(m, n, 1)
    A = _colmajor(a)
    Bm = _empty((ldb, nrhs), fmt)
    Bm[...] = array(0, fmt)._v
    Bm[:m] = b._v.reshape(m, nrhs)
    B = np.array(Bm.ravel(order="F"), order="C")
    rc = np.array(np.concatenate([array(r, fmt)._v.reshape(1) for r in rconds]), order="C")
    X = _empty(len(rc) * n * nrhs, fmt)
    S = _empty(max(min(m, n), 1), fmt)
    ranks = np.zeros(len(rc), dtype=np.intc)
    lib = _library(fmt, backend)
    info = lib.pb_gelss(*_cargs(fmt), m, n, nrhs, _ptr(fmt, A), _ptr(fmt, B), len(rc), _ptr(fmt, rc),
                        _ptr(fmt, X), _ptr(fmt, S), _lib.iptr(ranks))
    if info > 0:
        raise LinAlgError(f"SVD did not converge ({info} superdiagonals left)")
    if info < 0:
        raise RuntimeError(f"pfloat gelss failed ({info})")
    xs = [X[q * n * nrhs:(q + 1) * n * nrhs].reshape((n, nrhs), order="F") for q in range(len(rc))]
    xs = [PArray._wrap(x[:, 0] if b.ndim == 1 else x, fmt) for x in xs]
    return xs, [int(r) for r in ranks], PArray._wrap(S[:min(m, n)], fmt)


def lstsq(a, b, rcond=None, *, fmt=None, backend="auto"):
    """Least-squares solution of a x = b in a p-bit format, like numpy.linalg.lstsq.

    The solve is reference LAPACK 3.12.1 DGELSS (SVD-based, minimum-norm, singular values below
    rcond * s[0] treated as zero), ported so that every floating-point operation is correctly
    rounded in the format; in binary32/binary64 it is bit-identical to netlib SGELSS/DGELSS.

    Parameters
    ----------
    a : (M, N) PArray or array-like
    b : (M,) or (M, K) PArray or array-like
    rcond : float, optional
        Relative cutoff: singular values <= rcond * s[0] are treated as zero. Default: machine
        precision eps = 2^(1-p) (DGELSS's convention for a negative rcond). numpy's default,
        eps * max(M, N), is available by passing it; in a narrow format it can exceed 1 and
        discard the whole spectrum, so it is not the default here.
    fmt : Format, int or str, optional
        Required when neither a nor b is a PArray and no default is set (pfloat.precision).
    backend : "auto" or "emulator"
        "auto" runs exactly binary32/binary64 on native hardware (bit-identical, faster).

    Returns
    -------
    x : PArray (N,) or (N, K)
    residuals : PArray (K,) or (1,) -- sum of squared residuals computed in the format, when
        M > N and rank == N; otherwise empty
    rank : int
    s : PArray (min(M, N),) -- singular values, descending
    """
    a, b = _prepare(a, b, fmt)
    fmt = a.fmt
    m, n = a.shape
    rc = -1.0 if rcond is None else array(rcond, fmt)
    (x,), (rank,), s = _solve(a, b, [rc], backend)
    if m > n and rank == n:
        r = matmul(a, x) - b
        residuals = _sum(r * r, axis=0)
        residuals = residuals.reshape(1) if b.ndim == 1 else residuals
    else:
        residuals = PArray._wrap(np.zeros(0), fmt)
    return x, residuals, rank, s


def lstsq_cutoffs(a, b, rconds, *, fmt=None, backend="auto"):
    """DGELSS at several rcond values from one decomposition (each result is exactly what a
    separate lstsq call with that rcond returns). Returns ([x per rcond], [rank per rcond], s)."""
    a, b = _prepare(a, b, fmt)
    rc = [array(r, a.fmt) for r in rconds]
    return _solve(a, b, rc, backend)


# ---------------------------------------------------------------- square systems

def _operand(a, fmt) -> PArray:
    """a as a PArray: rounded into fmt when given, else in its own format (a PArray) or the
    default format."""
    if fmt is None and isinstance(a, PArray):
        return a
    return array(a, _resolve(fmt))


def _square(a, fmt, name) -> PArray:
    a = _operand(a, fmt)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name}: a must be a square 2-D array, got shape {a.shape}")
    return a


def _rhs(b: PArray, n: int) -> tuple[np.ndarray, int]:
    if b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError(f"b must have {n} rows")
    nrhs = 1 if b.ndim == 1 else b.shape[1]
    return _colmajor(b.reshape(n, nrhs)), nrhs


def lu_factor(a, *, fmt=None, backend="auto"):
    """LU factorization with partial pivoting, like scipy.linalg.lu_factor.

    Reference LAPACK DGETRF at block size 1 (the recursive DGETRF2: IDAMAX pivoting, DLASWP,
    DTRSM, DGEMM). a may be rectangular.

    Returns
    -------
    lu : PArray (M, N) -- U in the upper triangle, the unit-lower L below the diagonal
    piv : ndarray of int (min(M, N),) -- 0-based: row i was interchanged with row piv[i]

    Raises LinAlgError when a pivot is exactly zero (U is singular); scipy warns instead.
    """
    a = _operand(a, fmt)
    if a.ndim != 2:
        raise ValueError("a must be 2-D")
    fmt = a.fmt
    _check_lapack(fmt, "lu_factor")
    m, n = a.shape
    A = _colmajor(a)
    LU = _empty(m * n, fmt)
    ipiv = np.zeros(max(min(m, n), 1), dtype=np.intc)
    info = _library(fmt, backend).pb_getrf(*_cargs(fmt), m, n, _ptr(fmt, A), _ptr(fmt, LU), _lib.iptr(ipiv))
    if info < 0:
        raise RuntimeError(f"pfloat getrf failed ({info})")
    if info > 0:
        raise LinAlgError(f"singular matrix: U[{info - 1}, {info - 1}] is exactly zero")
    return _from_colmajor(LU, (m, n), fmt), ipiv[:min(m, n)].astype(np.int64) - 1


def lu_solve(lu_and_piv, b, *, backend="auto"):
    """Solve a x = b from lu_factor(a), like scipy.linalg.lu_solve (reference LAPACK DGETRS,
    TRANS = 'N')."""
    lu, piv = lu_and_piv
    if not isinstance(lu, PArray):
        raise TypeError("lu must be the PArray returned by lu_factor")
    fmt = lu.fmt
    n = lu.shape[0]
    if lu.ndim != 2 or lu.shape[1] != n:
        raise ValueError("lu_solve needs the factors of a square matrix")
    _, b = _pair(lu, b)
    B, nrhs = _rhs(b, n)
    ipiv = np.ascontiguousarray(np.asarray(piv, dtype=np.intc) + 1)
    LU, X = _colmajor(lu), _empty(n * nrhs, fmt)
    info = _library(fmt, backend).pb_getrs(*_cargs(fmt), n, nrhs, _ptr(fmt, LU), _lib.iptr(ipiv), _ptr(fmt, B),
                                           _ptr(fmt, X))
    if info != 0:
        raise RuntimeError(f"pfloat getrs failed ({info})")
    x = _from_colmajor(X, (n, nrhs), fmt)
    return x.reshape(n) if b.ndim == 1 else x


def solve(a, b, *, fmt=None, backend="auto"):
    """Solve the square system a x = b, like numpy.linalg.solve.

    Reference LAPACK DGESV: LU with partial pivoting (DGETRF at block size 1), then DGETRS.
    Raises LinAlgError when a pivot is exactly zero.
    """
    a, b = _prepare(a, b, fmt)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"solve: a must be square, got shape {a.shape}")
    return lu_solve(lu_factor(a, backend=backend), b, backend=backend)


def _potrf(a: PArray, lower: bool, backend: str) -> PArray:
    fmt = a.fmt
    _check_lapack(fmt, "cholesky")
    n = a.shape[0]
    A, C = _colmajor(a), _empty(n * n, fmt)
    info = _library(fmt, backend).pb_potrf(*_cargs(fmt), 0 if lower else 1, n, _ptr(fmt, A), _ptr(fmt, C))
    if info < 0:
        raise RuntimeError(f"pfloat potrf failed ({info})")
    if info > 0:
        raise LinAlgError(f"matrix is not positive definite (leading minor of order {info})")
    return _from_colmajor(C, (n, n), fmt)


def cho_factor(a, lower=False, *, fmt=None, backend="auto"):
    """Cholesky factorization for cho_solve, like scipy.linalg.cho_factor (reference LAPACK
    DPOTRF at block size 1, the recursive DPOTRF2).

    Only the lower (lower=True) or upper triangle of a is read. Returns (c, lower): c holds the
    factor in that triangle and the other triangle of a unchanged, as DPOTRF leaves it.
    """
    a = _square(a, fmt, "cho_factor")
    return _potrf(a, bool(lower), backend), bool(lower)


def cho_solve(c_and_lower, b, *, backend="auto"):
    """Solve a x = b from cho_factor(a), like scipy.linalg.cho_solve (reference LAPACK DPOTRS)."""
    c, lower = c_and_lower
    if not isinstance(c, PArray):
        raise TypeError("c must be the PArray returned by cho_factor")
    fmt = c.fmt
    n = c.shape[0]
    _, b = _pair(c, b)
    B, nrhs = _rhs(b, n)
    C, X = _colmajor(c), _empty(n * nrhs, fmt)
    info = _library(fmt, backend).pb_potrs(*_cargs(fmt), 0 if lower else 1, n, nrhs, _ptr(fmt, C), _ptr(fmt, B),
                                           _ptr(fmt, X))
    if info != 0:
        raise RuntimeError(f"pfloat potrs failed ({info})")
    x = _from_colmajor(X, (n, nrhs), fmt)
    return x.reshape(n) if b.ndim == 1 else x


def cholesky(a, *, upper=False, fmt=None, backend="auto"):
    """Cholesky factor of a symmetric positive-definite matrix, like numpy.linalg.cholesky:
    L with a = L L^T (default; only the lower triangle of a is read), or U with a = U^T U
    (upper=True; only the upper triangle is read). The other triangle of the result is zero.

    Reference LAPACK DPOTRF (UPLO = 'L' or 'U') at block size 1. Raises LinAlgError when a
    is not positive definite.
    """
    a = _square(a, fmt, "cholesky")
    c = _potrf(a, not upper, backend)
    n = a.shape[0]
    keep = np.triu(np.ones((n, n), dtype=bool)) if upper else np.tril(np.ones((n, n), dtype=bool))
    out = c._v.copy()
    out[~keep] = array(0, a.fmt)._v
    return PArray._wrap(out, a.fmt)


# ---------------------------------------------------------------- norms

def norm(x, axis=None, keepdims=False, *, fmt=None, backend="auto"):
    """Euclidean norm (Frobenius for a matrix with axis=None) by reference BLAS DNRM2.

    DNRM2 is Blue's algorithm: sums of squares accumulated in three scaled ranges, so it neither
    overflows nor underflows spuriously. With axis=None the entries are taken in C order (as
    scipy.linalg.norm does for the Frobenius norm); with an integer axis each vector along it is
    one DNRM2 call. Only the 2-norm is provided.
    """
    x = _operand(x, fmt)
    fmt = x.fmt
    _check_lapack(fmt, "norm")
    lib = _library(fmt, backend)
    if axis is None:
        rows = x._v.reshape(1, -1)
        lead, where = (), None
    else:
        axis = axis % x.ndim
        moved = np.moveaxis(x._v, axis, -1)
        rows = moved.reshape(-1, moved.shape[-1])
        lead, where = moved.shape[:-1], axis
    rows = np.ascontiguousarray(rows)
    out = _empty(rows.shape[0], fmt)
    if lib.pb_nrm2(*_cargs(fmt), rows.shape[0], rows.shape[1], _ptr(fmt, rows), _ptr(fmt, out)):
        raise RuntimeError("pfloat nrm2 failed")
    out = out.reshape(lead)
    if keepdims:
        out = out.reshape((1,) * x.ndim) if where is None else np.expand_dims(out, where)
    return PArray._wrap(out, fmt)
