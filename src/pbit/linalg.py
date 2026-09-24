"""Least squares in a p-bit format: reference LAPACK DGELSS, ported operation by operation."""
from __future__ import annotations

import numpy as np

from . import _lib
from ._array import PArray, _kernel, _pair, _resolve, array, matmul, sum as _sum
from ._formats import Format, as_format

__all__ = ["lstsq", "lstsq_cutoffs", "LinAlgError"]


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


def _solve(a: PArray, b: PArray, rconds, backend: str):
    fmt: Format = a.fmt
    m, n = a.shape
    nrhs = 1 if b.ndim == 1 else b.shape[1]
    ldb = max(m, n, 1)
    A = np.asfortranarray(a._v).ravel(order="F")
    B = np.zeros((ldb, nrhs), order="F")
    B[:m] = b._v.reshape(m, nrhs)
    B = B.ravel(order="F")
    rc = np.ascontiguousarray(rconds, dtype=np.float64)
    X = np.zeros(len(rc) * n * nrhs)
    S = np.zeros(max(min(m, n), 1))
    ranks = np.zeros(len(rc), dtype=np.intc)
    lib = _lib.library("emul") if backend == "emulator" else _kernel(fmt)
    info = lib.pb_gelss(_lib.ptr(fmt.constants()), m, n, nrhs, _lib.ptr(A), _lib.ptr(B), len(rc),
                        _lib.ptr(rc), _lib.ptr(X), _lib.ptr(S), _lib.iptr(ranks))
    if info > 0:
        raise LinAlgError(f"SVD did not converge ({info} superdiagonals left)")
    if info < 0:
        raise RuntimeError(f"pbit gelss failed ({info})")
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
        Required when neither a nor b is a PArray and no default is set (pbit.precision).
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
    rc = -1.0 if rcond is None else float(array(rcond, fmt))
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
    rc = [float(array(r, a.fmt)) for r in rconds]
    return _solve(a, b, rc, backend)
