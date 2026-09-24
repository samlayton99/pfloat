"""Netlib reference LAPACK 3.12.1 SGELSS / DGELSS, built from source, to check the port against.

The Fortran sources DGELSS and SGELSS reach are vendored in ``third_party/lapack-3.12.1`` of the
repository (with LAPACK's license); ``$PBIT_LAPACK_SRC`` can point elsewhere. They are compiled
with gfortran (``$FC``) and ``-ffp-contract=off`` (no fused multiply-add). One file is changed:
ILAENV returns block size 1, the unblocked configuration the port implements.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from .._lib import cache_dir

FFLAGS = ["-O2", "-ffp-contract=off", "-fPIC"]
FC = os.environ.get("FC", "gfortran")
_LIB = None


def source_dir() -> Path:
    if os.environ.get("PBIT_LAPACK_SRC"):
        return Path(os.environ["PBIT_LAPACK_SRC"])
    return Path(__file__).resolve().parents[3] / "third_party" / "lapack-3.12.1"


def available() -> bool:
    return shutil.which(FC) is not None and (source_dir() / "SRC" / "dgelss.f").exists()


def _undefined_fortran_symbols(obj: Path) -> list[str]:
    """Fortran routines an object file calls (nm prints '_name_' on macOS, 'U name_' on Linux)."""
    out = subprocess.run(["nm", "-u", str(obj)], check=True, capture_output=True, text=True).stdout
    names = []
    for line in out.splitlines():
        if not line.split():
            continue
        m = re.fullmatch(r"_?([a-z][a-z0-9_]*)_", line.split()[-1])
        if m:
            names.append(m.group(1))
    return names


def _find(src: Path, symbol: str) -> Path | None:
    for d in ("SRC", "BLAS/SRC", "INSTALL"):
        for ext in (".f", ".f90", ".F90", ".F"):
            path = src / d / f"{symbol}{ext}"
            if path.exists():
                return path
    return None


def build() -> Path:
    """Compile the SGELSS/DGELSS closure into a shared library (cached) and return its path."""
    src = source_dir()
    if not (src / "SRC" / "dgelss.f").exists():
        raise FileNotFoundError(f"reference LAPACK sources not found in {src}")
    digest = hashlib.sha256((" ".join(FFLAGS) + Path(__file__).read_text()).encode())
    for f in sorted(src.rglob("*.f*")):
        digest.update(f.read_bytes())
    build_dir = cache_dir() / f"reflapack_{digest.hexdigest()[:12]}"
    lib = build_dir / "libreflapack.so"
    if lib.exists():
        return lib
    build_dir.mkdir(parents=True, exist_ok=True)
    fc = [FC, *FFLAGS, "-J", str(build_dir), "-c"]
    objects = []
    for module in ("la_constants.f90", "la_xisnan.F90"):
        obj = build_dir / (module + ".o")
        subprocess.run([*fc, str(src / "SRC" / module), "-o", str(obj)], check=True)
        objects.append(obj)
    ilaenv = src / "SRC" / "ilaenv.f"
    text = ilaenv.read_text()
    anchor = "      GO TO ( 10, 10, 10, 80, 90, 100, 110, 120,"
    assert text.count(anchor) == 1
    patched = build_dir / "ilaenv.f"
    patched.write_text(text.replace(anchor, "      IF( ISPEC.EQ.1 ) THEN\n         ILAENV = 1\n"
                                            "         RETURN\n      END IF\n" + anchor))
    queue, seen = ["dgelss", "sgelss"], set()
    while queue:
        sym = queue.pop()
        if sym in seen:
            continue
        seen.add(sym)
        path = patched if sym == "ilaenv" else _find(src, sym)
        if path is None:
            continue
        obj = build_dir / f"{sym}.o"
        subprocess.run([*fc, str(path), "-o", str(obj)], check=True)
        objects.append(obj)
        queue += [s for s in _undefined_fortran_symbols(obj) if s not in seen]
    subprocess.run([FC, "-shared", "-o", str(lib), *map(str, objects)], check=True)
    return lib


def _load():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(build()))
    return _LIB


def gelss(a, b, rcond: float, dtype) -> dict:
    """Reference xGELSS (dtype float32 or float64) with the optimal workspace from a query.
    a: (M, N); b: (M,) or (M, K). Returns x (float64), singular values, rank, info."""
    lib = _load()
    double = np.dtype(dtype) == np.float64
    fn = lib.dgelss_ if double else lib.sgelss_
    ct = ctypes.c_double if double else ctypes.c_float
    a = np.asfortranarray(np.asarray(a, dtype=dtype)).copy(order="F")
    m, n = a.shape
    b = np.asarray(b, dtype=dtype)
    nrhs = 1 if b.ndim == 1 else b.shape[1]
    ldb = max(m, n, 1)
    bb = np.zeros((ldb, nrhs), dtype=dtype, order="F")
    bb[:m] = b.reshape(m, nrhs)
    s = np.zeros(max(min(m, n), 1), dtype=dtype)
    i = ctypes.c_int
    rank, info = i(0), i(0)
    P = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ct))  # noqa: E731

    def call(work, lwork):
        fn(ctypes.byref(i(m)), ctypes.byref(i(n)), ctypes.byref(i(nrhs)), P(a), ctypes.byref(i(max(m, 1))),
           P(bb), ctypes.byref(i(ldb)), P(s), ctypes.byref(ct(rcond)), ctypes.byref(rank), P(work),
           ctypes.byref(i(lwork)), ctypes.byref(info))

    query = np.zeros(1, dtype=dtype)
    call(query, -1)
    work = np.zeros(max(int(query[0]), 1), dtype=dtype)
    call(work, work.size)
    x = bb[:n].astype(np.float64)
    return {"x": x[:, 0] if b.ndim == 1 else x, "sigma": s[:min(m, n)].astype(np.float64),
            "rank": rank.value, "info": info.value}
