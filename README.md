# pfloat

[![tests](https://github.com/samlayton99/pfloat/actions/workflows/tests.yml/badge.svg)](https://github.com/samlayton99/pfloat/actions/workflows/tests.yml)

**p-bit floating point for Python: a correctly rounded array type for any significand width from 2 to 4096 bits, with reference LAPACK solvers (least squares, LU, Cholesky) ported to it.**

Pick a precision $p$. pfloat then guarantees three things:
- **Rounding.** Every $+$, $-$, $\times$, $\div$ and $\sqrt{\ }$ is rounded exactly as an IEEE 754 machine with a $p$-bit significand would round it: to nearest, ties to even, with subnormals and overflow.
- **Solvers.** Least squares (`DGELSS`), linear solves (`DGESV`) and Cholesky (`DPOTRF`) run reference LAPACK with every operation inside the solver at $p$ bits.
- **No hidden precision.** Nothing is computed in higher precision behind your back. Operations pfloat doesn't implement raise an error instead of silently falling back to binary64.

It's built for questions of the form "what does this computation do with $p$ bits of precision?": how error scales with $p$ (bit-complexity arguments), low-precision numerics, emulating fp16, bf16, tf32 or fp8 on a CPU, and running the same code in binary128 or at hundreds of bits.

```python
import numpy as np
import pfloat

A = np.random.default_rng(0).standard_normal((100, 5))
b = A @ np.arange(1.0, 6.0)

x, residuals, rank, s = pfloat.lstsq(A, b, fmt=pfloat.BF16)   # the whole solve in bfloat16 arithmetic
x, residuals, rank, s = pfloat.lstsq(A, b, fmt=19)            # ... with a 19-bit significand
x, residuals, rank, s = pfloat.lstsq(A, b, fmt=pfloat.FP128)  # ... or in IEEE binary128
x.to_numpy(rounding=True)                                     # the result, rounded to binary64
```

In binary32 and binary64 every solver is bit-identical to netlib reference LAPACK (`SGELSS`/`DGELSS`, `xGESV`, `xPOTRF`/`xPOTRS`, `xNRM2`). The other formats run the same code with every operation rounded to $p$ bits.

## Why

Most "low precision" experiments round the inputs and outputs but run the core of the computation, the solver above all, in binary64. The result then says little about $p$-bit arithmetic. pfloat makes the arithmetic model explicit and applies it everywhere, including inside a real, standard solver. For example, fitting a function by least squares with every step at $p$ bits gives an error that falls by one bit per bit of precision:

<p align="center"><img src="examples/precision_sweep.png" width="520" alt="error of a Chebyshev least-squares fit against working precision p"></p>

(`examples/precision_sweep.py`: 40 Chebyshev features built by recurrence at $p$ bits, solved with `pfloat.lstsq`, evaluated at $p$ bits; the error is measured in binary64.)

## Install

```bash
pip install pfloat              # formats up to 53 bits
pip install "pfloat[mp]"        # also formats above 53 bits (brings gmpy2, which bundles MPFR)
```

This needs Python 3.10+; numpy and mpmath are installed with it. The wheels for macOS (arm64, x86-64) and Linux (x86-64, aarch64) contain the compiled kernels. On other platforms pip builds them from the source distribution, which needs a C compiler.

Formats above 53 bits run on [MPFR](https://www.mpfr.org). Their kernels are compiled the first time you use them, in a few seconds, into `~/.cache/pfloat` (or `$PFLOAT_CACHE_DIR`). That needs a C++ compiler (`c++`, or `$CXX`) and MPFR, found in this order:
1. `$PFLOAT_MPFR_PREFIX`;
2. Homebrew or the system (`brew install mpfr`, `apt install libmpfr-dev`);
3. the copy bundled in gmpy2.

Tested on macOS (arm64, clang) and Linux (x86-64, gcc). The reference-LAPACK and audit checks run in CI on both.

## Formats

```python
pfloat.Format(19)            # 19-bit significand, exponent range [-958, 959] (binary64's, minus 64 binades each end)
pfloat.Format(11, emax=15)   # IEEE-style range, emin = 1 - emax  ->  same as pfloat.FP16
pfloat.Format(1000)          # 1000-bit significand, binary128's exponent range [-16382, 16383]
pfloat.FP8_E5M2, pfloat.BF16, pfloat.FP16, pfloat.TF32, pfloat.FP32, pfloat.FP64, pfloat.FP128, pfloat.FP256
```

`fmt.eps` is $2^{1-p}$ and `fmt.unit_roundoff` is $2^{-p}$. `fmt.max`, `fmt.min_normal` and `fmt.min_subnormal` give the range. All of these are exact `Fraction`s.

**How many bits.** $p$ runs from 2 to 4096, on two backends that implement the same arithmetic:

| $p$ | backend | cost per operation |
|---|---|---|
| 2 to 53 | Emulated exactly on binary64 hardware. FP32 and FP64 themselves run natively. | About 10 to 15 times native |
| 54 to 4096 | MPFR. Each value is a fixed-size record of $\lceil p/64\rceil$ 64-bit words, rounded up to a power of two. | About 10 times the emulator at 113 bits and 13 to 19 times at 237, growing with $p$ |

The limit of 4096 bits is a chosen cap, not a hardware one. The two backends are checked against each other: the MPFR build, run at $p \le 53$, gives the emulator's results bit for bit.

## The array type

```python
x = pfloat.array([0.1, 0.2, 0.3], fmt=12)     # values rounded into the format once
y = x * 3 + 1                                # every operation correctly rounded at 12 bits
pfloat.sqrt(x), pfloat.tanh(x)               # sqrt correctly rounded; tanh from +,-,*,/ (<= 3 ulp)
x.sum(), x @ x, A @ x                        # fixed orders: left to right; reference BLAS for products
np.add(x, 1), np.sum(x)                      # numpy calls pfloat implements run in the format
np.exp(x)                                    # TypeError: not implemented at p-bit precision
with pfloat.precision(pfloat.BF16):          # a default format for a block of code
    z = pfloat.array(data)
```

A `PArray` holds values of one format. Plain numbers and numpy arrays used with it are rounded into the format first, the way a $p$-bit machine loads them. Mixing two formats raises; convert explicitly with `.astype(fmt)`.

| from | into pfloat | out of pfloat |
|---|---|---|
| numpy (float16/32/64, ints, bool, longdouble) | `pfloat.array(a, fmt)` | `x.to_numpy()` (binary64; raises unless exact, which it always is up to 53 bits); `x.to_numpy(rounding=True)`; `np.asarray(x)` |
| Python numbers and lists | `pfloat.array(1/3, fmt)` | `float(x)`, `x.tolist()` |
| exact numbers: `int`, `Fraction`, `Decimal`, mpmath `mpf`, gmpy2 `mpfr` | rounded once from the exact value (no detour through binary64) | `x.to_fractions()`, `x.to_mpmath()` |
| torch tensors (any float dtype, any device) | `pfloat.array(t, fmt)` | `x.to_torch()` (float64); `x.to_torch(torch.bfloat16)` (only if exact) |
| another format | `x.astype(fmt)` | |

## Linear algebra

Each function is a reference LAPACK 3.12.1 routine, ported operation by operation, with the machine parameters of the format:

| function | like | reference LAPACK |
|---|---|---|
| `pfloat.lstsq(a, b, rcond=None)` | `numpy.linalg.lstsq` | `DGELSS`: SVD-based, minimum norm, rank-revealing |
| `pfloat.linalg.lstsq_cutoffs(a, b, rconds)` | | `DGELSS` at several cutoffs from one decomposition |
| `pfloat.solve(a, b)` | `numpy.linalg.solve` | `DGESV`: LU with partial pivoting |
| `pfloat.linalg.lu_factor(a)`, `lu_solve((lu, piv), b)` | `scipy.linalg` | `DGETRF` (recursive `DGETRF2`), `DGETRS` |
| `pfloat.cholesky(a, upper=False)` | `numpy.linalg.cholesky` | `DPOTRF` (recursive `DPOTRF2`) |
| `pfloat.linalg.cho_factor(a, lower=False)`, `cho_solve((c, lower), b)` | `scipy.linalg` | `DPOTRF`, `DPOTRS` |
| `pfloat.norm(x, axis=None)` | `numpy.linalg.norm` (2-norm) | `DNRM2`: Blue's algorithm, safe from overflow and underflow |

`lstsq` covers every `DGELSS` path:
- over- and underdetermined systems, with and without the QR/LQ step;
- several right-hand sides;
- rescaling of badly scaled data;
- rank deficiency.

Its default `rcond` is machine precision $2^{1-p}$ (LAPACK's convention). numpy's `eps * max(M, N)` can exceed 1 in a narrow format.

A singular matrix (an exactly zero pivot) in `solve` or `lu_factor`, and a matrix that isn't positive definite in `cholesky`, raise `pfloat.LinAlgError`. Every function accepts `backend="emulator"`, which runs FP32 and FP64 through the emulator instead of the native build; the results are the same bits.

## Guarantees, and how they are checked

- **Arithmetic.** Each $+,-,\times,\div,\sqrt{\ }$ equals MPFR's correctly rounded result:
  - for every $p$ from 2 to 53, and at $p$ = 54, 64, 113, 200, 237 and 1000;
  - including midpoint ties, subnormals, overflow and results near $2^{-1000}$.

  FP32, FP64 and FP16 also equal numpy's hardware arithmetic.
- **Solvers.** In FP32 and FP64, each solver is bit-identical to netlib reference LAPACK, built from the vendored Fortran with gfortran and no fused multiply-add:
  - `lstsq` against `SGELSS`/`DGELSS` on every path: random, rank-deficient and graded matrices, extreme scales, several cutoffs;
  - `solve`, `lu_factor` and `lu_solve` against `xGESV`/`xGETRF`: square and rectangular, with pivots in the subnormal range;
  - `cholesky`, `cho_factor` and `cho_solve` against `xPOTRF`/`xPOTRS`: both triangles, failures included;
  - `norm` against `xNRM2`, at magnitudes from the underflow threshold to near overflow.
- **Backends agree.** The native FP32/FP64 builds, the emulator and the MPFR build (run at $p \le 53$) give identical bits through every solver.
- **No hidden precision.** A static audit (`python -m pfloat.audit`) parses the C with clang, with all macros expanded and all branches included. It finds no floating-point operation outside the rounding layer, with one documented integer-valued exception from LAPACK. Negative controls show it catches planted leaks.
- **Threads.** Results are bit-identical for any thread count.

The details are in [docs/DESIGN.md](docs/DESIGN.md): how the emulator gets the exact rounding, the MPFR backend, the tanh algorithm, the ports, and the verification.

## Performance

Dense Gaussian matrices on an Apple M-series CPU with 10 cores (`python benchmarks/bench.py`).

`pfloat.lstsq`:

| problem | format | 1 thread | 10 threads | numpy float64 lstsq |
|---|---|---:|---:|---:|
| 1000 x 200 | p = 24 (emulated) | 0.29 s | 0.13 s | 0.005 s |
| 1000 x 200 | p = 40 (emulated) | 0.30 s | 0.14 s | 0.005 s |
| 1000 x 200 | fp32 (native path) | 0.03 s | 0.03 s | 0.005 s |
| 1000 x 200 | fp64 (native path) | 0.03 s | 0.03 s | 0.005 s |
| 1000 x 200 | fp128 (MPFR) | 3.11 s | 0.84 s | 0.005 s |
| 1000 x 200 | fp256 (MPFR) | 5.36 s | 1.42 s | 0.005 s |
| 2401 x 513 | p = 24 (emulated) | 4.55 s | 1.09 s | 0.028 s |
| 2401 x 513 | p = 40 (emulated) | 4.69 s | 1.13 s | 0.028 s |
| 2401 x 513 | fp32 (native path) | 0.39 s | 0.24 s | 0.028 s |
| 2401 x 513 | fp64 (native path) | 0.41 s | 0.26 s | 0.028 s |
| 2401 x 513 | fp128 (MPFR) | 50.53 s | 8.86 s | 0.028 s |
| 2401 x 513 | fp256 (MPFR) | 86.21 s | 15.28 s | 0.028 s |
| 4801 x 1025 | p = 24 (emulated) | 35.07 s | 6.78 s | 0.154 s |
| 4801 x 1025 | p = 40 (emulated) | 36.28 s | 7.02 s | 0.154 s |
| 4801 x 1025 | fp32 (native path) | 3.18 s | 1.11 s | 0.154 s |
| 4801 x 1025 | fp64 (native path) | 3.30 s | 1.21 s | 0.154 s |

`pfloat.solve` and `pfloat.cholesky`:

| matrix | format | solve, 1 thread | solve, 10 threads | cholesky, 1 thread | cholesky, 10 threads |
|---|---|---:|---:|---:|---:|
| 500 x 500 | p = 24 (emulated) | 0.20 s | 0.04 s | 0.05 s | 0.01 s |
| 500 x 500 | p = 40 (emulated) | 0.20 s | 0.04 s | 0.05 s | 0.01 s |
| 500 x 500 | fp32 (native path) | 0.01 s | 0.01 s | 0.01 s | 0.01 s |
| 500 x 500 | fp64 (native path) | 0.01 s | 0.01 s | 0.01 s | 0.01 s |
| 500 x 500 | fp128 (MPFR) | 1.53 s | 0.29 s | 0.73 s | 0.14 s |
| 500 x 500 | fp256 (MPFR) | 2.51 s | 0.48 s | 1.15 s | 0.21 s |
| 1000 x 1000 | p = 24 (emulated) | 1.57 s | 0.28 s | 0.37 s | 0.09 s |
| 1000 x 1000 | p = 40 (emulated) | 1.57 s | 0.28 s | 0.37 s | 0.09 s |
| 1000 x 1000 | fp32 (native path) | 0.10 s | 0.03 s | 0.05 s | 0.03 s |
| 1000 x 1000 | fp64 (native path) | 0.10 s | 0.03 s | 0.05 s | 0.03 s |
| 1000 x 1000 | fp128 (MPFR) | 12.20 s | 2.07 s | 5.75 s | 1.00 s |
| 1000 x 1000 | fp256 (MPFR) | 19.83 s | 3.33 s | 9.09 s | 1.51 s |

numpy's float64 `solve` and `cholesky` take 1 ms and 0.3 ms at 500 x 500, and 5 ms and 2 ms at 1000 x 1000.

Exactly FP32 and FP64 run on native hardware through the same port. This path is bit-identical to the emulator. Loops over independent columns and rows run on all cores; set the count with `pfloat.set_num_threads(n)` or `PFLOAT_NUM_THREADS`. The results don't depend on it.

## Limits

- **Round to nearest, ties to even, only.** Other rounding modes and IEEE exception flags aren't emulated. `pfloat.events()` counts overflows, underflows and non-finite results in formats up to 53 bits.
- **Exponent ranges.** Ranges are IEEE-style ($e_{\min} = 1 - e_{\max}$).
  - Up to 53 bits: $e_{\max} \le 1023$, and $e_{\min} \ge -1022 + p + 1$ except for exactly binary64.
  - Above 53 bits: $e_{\max} \le 2^{30}$.

  The non-IEEE OCP FP8 E4M3 isn't provided.
- **Narrow formats.** In a format whose exponent range is narrow relative to $p$ (FP16, FP8), LAPACK's safe-scaling branches are taken often and cost accuracy. That is how LAPACK's algorithm behaves in such a format. `Format(p)` isolates the effect of significand width.
- **Bit-identical to reference LAPACK only.** The match is with reference LAPACK in its unblocked configuration, not with optimized libraries (OpenBLAS, MKL, Accelerate), which reorder operations.
- **Elementary functions.** Only sqrt and tanh; others raise.
- **Platforms.** macOS and Linux; Windows is untested.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
pytest                   # needs gmpy2; the reference-LAPACK tests also need gfortran, the audit needs clang
python -m pfloat.audit   # prints the audit findings (only the documented exception)
```

Releases are built by `.github/workflows/release.yml`: wheels for macOS and Linux, each tested, published to PyPI by trusted publishing when a GitHub release is published.

## License

BSD 3-Clause. `src/pfloat/csrc/lapack_gelss.h` and `lapack_solve.h` are derived from Reference LAPACK, and `third_party/lapack-3.12.1` vendors the netlib Fortran used for verification. Both are under LAPACK's modified BSD license, in `third_party/lapack-3.12.1/LICENSE`.
