"""Build the pfloat kernels at install time.

The kernels are plain C shared libraries loaded with ctypes (they define no Python module). They
are compiled with strict floating-point flags: no fused multiply-add (-ffp-contract=off), no
fast-math, C11; the native binary32/binary64 builds also disable auto-vectorization, so every
loop runs its scalar operations as written.
"""
import hashlib
import os
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

STRICT = ["-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-pthread"]
CSRC = os.path.join("src", "pfloat", "csrc")
SOURCES = ("emul.c", "native.c", "native64.c", "native32.c", "kernels.h", "lapack_gelss.h",
           "lapack_solve.h")


def sources_digest() -> str:
    """The same digest pfloat._lib computes at run time: prebuilt kernels are used only when it matches."""
    h = hashlib.sha256()
    for name in SOURCES:
        with open(os.path.join(CSRC, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


class StrictBuildExt(build_ext):
    def build_extensions(self):
        cc = self.compiler.compiler_so[0] if getattr(self.compiler, "compiler_so", None) else "cc"
        try:
            version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.lower()
        except OSError:
            version = ""
        novec = (["-fno-vectorize", "-fno-slp-vectorize"] if "clang" in version
                 else ["-fno-tree-vectorize", "-fno-tree-slp-vectorize"])
        for ext in self.extensions:
            ext.extra_compile_args = STRICT + (novec if ext.name != "pfloat._emul" else [])
            ext.extra_link_args = ["-pthread", "-lm"]
        super().build_extensions()

    def run(self):
        super().run()
        target = os.path.join("src" if self.inplace else self.build_lib, "pfloat", "_kernels.sha256")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            f.write(sources_digest() + "\n")

    def get_export_symbols(self, ext):
        return []  # ctypes libraries: no PyInit_ function to export


def kernel(name, source):
    return Extension(f"pfloat.{name}", sources=[f"src/pfloat/csrc/{source}"],
                     include_dirs=["src/pfloat/csrc"],
                     depends=[f"src/pfloat/csrc/{h}" for h in ("kernels.h", "lapack_gelss.h", "lapack_solve.h", "native.c")])


setup(ext_modules=[kernel("_emul", "emul.c"), kernel("_f64", "native64.c"), kernel("_f32", "native32.c")],
      cmdclass={"build_ext": StrictBuildExt})
