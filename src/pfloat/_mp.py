"""Extended precision (p > 53): the kernels built on MPFR, and exact value encoding.

A value of a format with p > 53 is stored as a fixed-size record (kind, sign, MPFR exponent,
ceil(p/64) significand limbs, rounded up to a power of two), the layout of ``T`` in
``csrc/mp.cpp``. The C++ build is compiled on first use against MPFR, found at
``$PFLOAT_MPFR_PREFIX``, then Homebrew / system locations, then the copy bundled in the gmpy2
wheel (``pip install pfloat[mp]``).
"""
from __future__ import annotations

import ctypes
from fractions import Fraction
import glob
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

from . import _lib

MAX_P = 4096
_LIBS: dict[int, ctypes.CDLL] = {}
_LOCK = threading.Lock()
_v = ctypes.c_void_p


def limbs_for(p: int) -> int:
    """Significand limbs per value: ceil(p / 64) rounded up to a power of two."""
    need = max(1, -(-p // 64))
    n = 1
    while n < need:
        n *= 2
    return n


def dtype(nlimbs: int) -> np.dtype:
    return np.dtype([("kind", "<i4"), ("sign", "<i4"), ("exp", "<i8"), ("limbs", "<u8", (nlimbs,))])


# ---------------------------------------------------------------- finding MPFR

def _mpfr_candidates():
    env = os.environ.get("PFLOAT_MPFR_PREFIX")
    if env:
        yield Path(env) / "include", Path(env) / "lib", None
    for mpfr, gmp in (("/opt/homebrew/opt/mpfr", "/opt/homebrew/opt/gmp"), ("/usr/local/opt/mpfr", "/usr/local/opt/gmp"),
                      ("/opt/homebrew", "/opt/homebrew"), ("/usr/local", "/usr/local"), ("/usr", "/usr")):
        yield (Path(mpfr) / "include", Path(mpfr) / "lib", Path(gmp) / "include")
    try:
        import gmpy2
        pkg = Path(gmpy2.__file__).parent
        yield pkg, pkg.parent / "gmpy2.libs", pkg
    except ImportError:
        pass


def _find_mpfr():
    """(include dirs, link arguments, libraries to repoint on macOS) for MPFR and GMP."""
    for inc, libdir, gmp_inc in _mpfr_candidates():
        if not (inc / "mpfr.h").exists() or not libdir.exists():
            continue
        incs = [inc] + ([gmp_inc] if gmp_inc and gmp_inc != inc else [])
        if not any((d / "gmp.h").exists() for d in incs):
            for extra in (Path("/opt/homebrew/opt/gmp/include"), Path("/usr/local/opt/gmp/include"), Path("/usr/include"),
                          Path("/usr/include") / (os.uname().machine + "-linux-gnu")):
                if (extra / "gmp.h").exists():
                    incs.append(extra)
                    break
        libs = {}
        for name in ("mpfr", "gmp"):
            found = sorted(glob.glob(str(libdir / f"lib{name}*.dylib")) + glob.glob(str(libdir / f"lib{name}*.so*")))
            if not found and name == "gmp":
                for d in (Path("/opt/homebrew/opt/gmp/lib"), Path("/usr/local/opt/gmp/lib"), libdir):
                    found = sorted(glob.glob(str(d / "libgmp*.dylib")) + glob.glob(str(d / "libgmp*.so*")))
                    if found:
                        break
            if found:
                libs[name] = Path(found[0])
        if "mpfr" in libs and "gmp" in libs:
            return incs, libs
    raise RuntimeError("formats with p > 53 need MPFR: install it (brew install mpfr / apt install libmpfr-dev), "
                       "or `pip install pfloat[mp]` (gmpy2 bundles MPFR), or set $PFLOAT_MPFR_PREFIX")


def library(nlimbs: int) -> ctypes.CDLL:
    """The extended-precision kernels for values of nlimbs 64-bit limbs."""
    if nlimbs in _LIBS:
        return _LIBS[nlimbs]
    with _LOCK:
        if nlimbs in _LIBS:
            return _LIBS[nlimbs]
        cxx = os.environ.get("CXX", "c++")
        if shutil.which(cxx) is None:
            raise RuntimeError(f"formats with p > 53 are compiled on first use; C++ compiler '{cxx}' not found (set $CXX)")
        incs, libs = _find_mpfr()
        version = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()
        flags = ["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared", "-pthread",
                 f"-DPB_LIMBS={nlimbs}"]
        digest = hashlib.sha256()
        for name in ("mp.cpp", "kernels.h", "lapack_gelss.h", "lapack_solve.h"):
            digest.update((_lib.CSRC / name).read_bytes())
        digest.update(" ".join(flags + [str(p) for p in libs.values()] + version[:1] + [sys.platform]).encode())
        out = _lib.cache_dir() / f"pfloat_mp{nlimbs}_{digest.hexdigest()[:16]}.so"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
                target = Path(tmp) / out.name
                rpaths = sorted({str(p.parent) for p in libs.values()})
                # room in the Mach-O header for the load commands repointed below (long paths)
                pad = ["-Wl,-headerpad_max_install_names"] if sys.platform == "darwin" else []
                cmd = [cxx, *flags, "-I", str(_lib.CSRC), *sum((["-I", str(d)] for d in incs), []),
                       str(_lib.CSRC / "mp.cpp"), str(libs["mpfr"]), str(libs["gmp"]),
                       *[f"-Wl,-rpath,{r}" for r in rpaths], *pad, "-o", str(target)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f"building the pfloat extended-precision kernels failed:\n{' '.join(cmd)}\n{result.stderr}")
                if sys.platform == "darwin":  # point the load commands at the actual files (gmpy2's use placeholder names)
                    listed = subprocess.run(["otool", "-L", str(target)], capture_output=True, text=True).stdout
                    for line in listed.splitlines()[1:]:
                        ref = line.strip().split(" (")[0]
                        for name, path in libs.items():
                            if f"lib{name}" in Path(ref).name and ref != str(path):
                                subprocess.run(["install_name_tool", "-change", ref, str(path), str(target)], check=True)
                    subprocess.run(["codesign", "--force", "-s", "-", str(target)], capture_output=True)
                os.replace(target, out)
        lib = ctypes.CDLL(str(out))
        d, i, z, c = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_int
        lib.pb_binary.argtypes = [d, _v, c, z, _v, _v, _v]
        lib.pb_unary.argtypes = [d, _v, c, z, _v, _v]
        lib.pb_sum.argtypes = [d, _v, z, z, _v, _v]
        lib.pb_matmul.argtypes = [d, _v, z, z, z, _v, _v, _v]
        lib.pb_gelss.argtypes = [d, _v, c, c, c, _v, _v, c, _v, _v, _v, i]
        lib.pb_getrf.argtypes = [d, _v, c, c, _v, _v, i]
        lib.pb_getrs.argtypes = [d, _v, c, c, _v, i, _v, _v]
        lib.pb_potrf.argtypes = [d, _v, c, c, _v, _v]
        lib.pb_potrs.argtypes = [d, _v, c, c, c, _v, _v, _v]
        lib.pb_nrm2.argtypes = [d, _v, z, z, _v, _v]
        lib.pb_from_double.argtypes = [d, z, d, _v]
        lib.pb_to_double.argtypes = [d, z, _v, d, ctypes.POINTER(ctypes.c_long)]
        lib.pb_compare.argtypes = [d, c, z, _v, _v, _v]
        lib.pb_set_threads.argtypes = [c]
        lib.pb_set_threads(_lib.get_num_threads())
        assert lib.pb_limbs() == nlimbs
        _lib.register_extra(lib)
        _LIBS[nlimbs] = lib
        return lib


def vptr(a: np.ndarray):
    """A pointer to the records that keeps the array alive (a temporary may be passed)."""
    return a.ctypes.data_as(_v)


# ---------------------------------------------------------------- exact encoding

def encode(values, p: int, nlimbs: int) -> np.ndarray:
    """Pack already-rounded values (Fraction, or float 0/inf/nan/signed zero) into records."""
    flat = list(values)
    out = np.zeros(len(flat), dtype=dtype(nlimbs))
    n = -(-p // 64)
    mask = (1 << 64) - 1
    for i, v in enumerate(flat):
        if isinstance(v, float) and v != v:
            out[i]["kind"] = 3
            continue
        if isinstance(v, float) and v in (float("inf"), float("-inf")):
            out[i]["kind"], out[i]["sign"] = 2, int(v < 0)
            continue
        if v == 0:
            out[i]["sign"] = int(isinstance(v, float) and str(v).startswith("-"))
            continue
        v = Fraction(v)
        sign, a = int(v < 0), abs(v)
        num, den = a.numerator, a.denominator
        assert den & (den - 1) == 0, "encode takes values of the format (dyadic rationals)"
        z = -(den.bit_length() - 1)
        tz = (num & -num).bit_length() - 1   # trailing zero bits belong to the exponent
        num >>= tz
        z += tz
        bl = num.bit_length()
        assert bl <= p, "value has more than p significant bits"
        sig = num << (64 * n - bl)
        out[i]["kind"], out[i]["sign"], out[i]["exp"] = 1, sign, z + bl
        out[i]["limbs"][:n] = [(sig >> (64 * k)) & mask for k in range(n)]
    return out


def decode(records: np.ndarray, p: int):
    """Exact values of records: Fractions (and floats for zero signs, infinities, NaN)."""
    n = -(-p // 64)
    vals = []
    for r in records.reshape(-1):
        kind, sign = int(r["kind"]), int(r["sign"])
        if kind == 3:
            vals.append(float("nan"))
        elif kind == 2:
            vals.append(float("-inf") if sign else float("inf"))
        elif kind == 0:
            vals.append(-0.0 if sign else 0.0)
        else:
            sig = sum(int(r["limbs"][k]) << (64 * k) for k in range(n))
            v = Fraction(sig) * Fraction(2) ** (int(r["exp"]) - 64 * n)
            vals.append(-v if sign else v)
    return vals
