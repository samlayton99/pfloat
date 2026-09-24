"""Build and load the C kernels.

The C sources in ``csrc/`` are compiled on first use with the system C compiler (``$CC``, else
``cc``) into a cache directory (``$PBIT_CACHE_DIR``, else ``$XDG_CACHE_HOME/pbit``, else
``~/.cache/pbit``), keyed by the sources, the flags and the compiler version. Three libraries:

- ``emul``: the p-bit emulator (any format);
- ``f64`` and ``f32``: the same kernels on native binary64 / binary32, used as a fast path for
  exactly those formats (bit-identical to the emulator; checked in the tests).

Every build uses ``-ffp-contract=off`` (no fused multiply-add) and no fast-math. The native
builds also disable auto-vectorization, so each loop runs its scalar operations as written.
Loops over independent items (columns, rows, elements) run on ``set_num_threads`` threads; each
result keeps its own operation order, so the output is bit-identical for any thread count.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

CSRC = Path(__file__).resolve().parent / "csrc"
SOURCES = ("emul.c", "native.c", "kernels.h", "lapack_gelss.h")
BASE_FLAGS = ["-O2", "-std=c11", "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared", "-pthread"]
_LIBS: dict[str, ctypes.CDLL] = {}
_LOCK = threading.Lock()
_THREADS = [int(os.environ.get("PBIT_NUM_THREADS", 0)) or (os.cpu_count() or 1)]

_d = ctypes.POINTER(ctypes.c_double)
_i = ctypes.POINTER(ctypes.c_int)
_z = ctypes.c_size_t
_c = ctypes.c_int


def cache_dir() -> Path:
    if os.environ.get("PBIT_CACHE_DIR"):
        return Path(os.environ["PBIT_CACHE_DIR"])
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "pbit"


def _compiler() -> tuple[str, str]:
    cc = os.environ.get("CC", "cc")
    if shutil.which(cc) is None:
        raise RuntimeError(f"pbit needs a C compiler to build its kernels on first use; '{cc}' was not "
                           "found (set $CC)")
    version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()
    return cc, version[0] if version else cc


def _flags(kind: str, version: str) -> list[str]:
    flags = list(BASE_FLAGS)
    if kind != "emul":
        flags.append(f"-DPB_TYPE={1 if kind == 'f64' else 2}")
        flags += (["-fno-vectorize", "-fno-slp-vectorize"] if "clang" in version.lower()
                  else ["-fno-tree-vectorize", "-fno-tree-slp-vectorize"])
    return flags


def library(kind: str) -> ctypes.CDLL:
    """The compiled kernels: kind is 'emul', 'f64' or 'f32'."""
    if kind in _LIBS:
        return _LIBS[kind]
    with _LOCK:
        if kind in _LIBS:
            return _LIBS[kind]
        if kind not in ("emul", "f64", "f32"):
            raise ValueError(kind)
        cc, version = _compiler()
        flags = _flags(kind, version)
        source = CSRC / ("emul.c" if kind == "emul" else "native.c")
        digest = hashlib.sha256()
        for name in SOURCES:
            digest.update((CSRC / name).read_bytes())
        digest.update(" ".join(flags).encode() + version.encode() + sys.platform.encode())
        out = cache_dir() / f"pbit_{kind}_{digest.hexdigest()[:16]}.so"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
                target = Path(tmp) / out.name
                cmd = [cc, *flags, "-I", str(CSRC), str(source), "-o", str(target), "-lm"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f"building the pbit {kind} kernels failed:\n{' '.join(cmd)}\n{result.stderr}")
                os.replace(target, out)
        lib = ctypes.CDLL(str(out))
        lib.pb_binary.argtypes = [_d, _c, _z, _d, _d, _d]
        lib.pb_unary.argtypes = [_d, _c, _z, _d, _d]
        lib.pb_sum.argtypes = [_d, _z, _z, _d, _d]
        lib.pb_matmul.argtypes = [_d, _z, _z, _z, _d, _d, _d]
        lib.pb_gelss.argtypes = [_d, _c, _c, _c, _d, _d, _c, _d, _d, _d, _i]
        lib.pb_set_threads.argtypes = [_c]
        lib.pb_set_threads(_THREADS[0])
        if kind == "emul":
            lib.pb_round.argtypes = [_d, _z, _d, _d]
            lib.pb_events.argtypes = [ctypes.POINTER(ctypes.c_long)]
        _LIBS[kind] = lib
        return lib


def set_num_threads(n: int) -> int:
    """Threads used by the kernels (results are bit-identical for any count). Returns the value set."""
    n = max(1, int(n))
    _THREADS[0] = n
    for lib in _LIBS.values():
        lib.pb_set_threads(n)
    return n


def get_num_threads() -> int:
    return _THREADS[0]


def ptr(a):
    return a.ctypes.data_as(_d)


def iptr(a):
    return a.ctypes.data_as(_i)
