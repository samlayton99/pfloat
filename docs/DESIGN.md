# pfloat design

This document explains how pfloat computes, why the results are exactly the p-bit results, and how that is checked.

## 1. The arithmetic model

A format $(p, e_{\min}, e_{\max})$ has a $p$-bit significand ($2\le p\le 4096$) and IEEE-style exponents ($e_{\min} = 1 - e_{\max}$). Its finite values are $\pm m\,2^{e-p+1}$ with integer $m<2^p$ and $e\ge e_{\min}$, plus signed zeros, infinities and NaN. The five operations $+, -, \times, \div, \sqrt{\ }$ return the exact result rounded to the nearest format value, ties to even, with gradual underflow and overflow to infinity. This is exactly IEEE 754's definition, applied to a significand of any width.

Nothing is fused: $a\times b + c$ is two roundings. Every other operation is either exact (negation, absolute value, comparison, copying a sign, scaling by a power of two that stays in range, rounding to an integer) or built from the five. Constants that an algorithm needs, such as $1/\ln 2$, the Taylor coefficients of $e^x$, or LAPACK's machine parameters, are computed once at high precision and stored after one correct rounding into the format, the same way a p-bit machine's library would store them.

Two backends implement the model. Formats with $p \le 53$ are emulated on binary64 hardware (section 2); FP32 and FP64 themselves also run natively. Formats with $p > 53$ run on MPFR (section 3). The kernels and the LAPACK ports are one C source, compiled against each backend's rounding macros.

## 2. How the emulator gets the exact rounding ($p \le 53$)

The carrier is binary64. For $s = a \circ b$ computed in binary64 and the exact result $t$:

- **Rounding $s$ is the same as rounding $t$, except when $s$ lies exactly on a rounding midpoint of the format.** Midpoints are binary64 numbers (the format has at most 52 bits below its leading one, so a midpoint needs at most 53), and $t$ is within half a binary64 ulp of $s$. So $t$ cannot be on the other side of a midpoint unless $s$ is the midpoint itself. The same holds at binade boundaries and in the subnormal range, where the format's quantum is coarser still.
- **Only at a midpoint does the emulator need to know which side of $s$ the exact result $t$ lies on.** It computes that side exactly with an error-free transform:
  - TwoSum for $\pm$;
  - an FMA residual for $\times$ ($ab - s$), $\div$ ($a - qb$) and $\sqrt{\ }$ ($a - s^2$). These residuals are exactly representable whenever they do not underflow. For results below $2^{-900}$ they are formed on operands scaled by an exact power of two ($2^{256}$, or $2^{512}$ under the root), so they never underflow.

  The FMA is only used to learn this sign; no value of the computation is ever fused.
- **Why the carrier must represent the format.** The format's values, subnormals included, must all be normal binary64 numbers. That requires $e_{\min} \ge -1022 + p + 1$, unless the format is binary64 itself, which is the hardware. The default range $[-958, 959]$ satisfies this for every $p$.

**The fast path.** The common case is a normal result, not on a midpoint, that doesn't overflow. There the rounding is one integer add and one mask on the binary64 bit pattern: `(mag + half) & ~mask`. Everything else goes to a general routine: zeros, midpoints, subnormal results, overflow, infinities and NaN.

## 3. The MPFR backend ($p > 53$)

A value is a fixed-size record: a kind (zero, finite, infinity, NaN), a sign, MPFR's exponent, and $\lceil p/64\rceil$ 64-bit significand words, rounded up to a power of two (so one build serves every $p$ up to its width: 64, 128, 256, ... 4096 bits). numpy stores these records as a structured array. MPFR operates on them in place through its custom interface, so no value is allocated per operation.

Each of the five operations is one MPFR call at precision $p$, rounding to nearest, followed by `mpfr_subnormalize`. MPFR's exponent range is set to the format's: $e_{\min} - p + 2$ to $e_{\max} + 1$ in MPFR's convention, where a value is $0.d_1d_2\ldots \times 2^{e}$. This is the recipe MPFR's manual gives for emulating an IEEE format. `mpfr_subnormalize` uses the ternary value of the first rounding, so a subnormal result is rounded once, correctly, not twice.

Values enter and leave exactly:
- Inputs are rounded from their exact rational value.
- Outputs are read back as exact `Fraction`s.
- `to_numpy()` raises unless every value is a binary64 number; `to_numpy(rounding=True)` rounds.

Constants are stored as exact records, not through binary64. The build is compiled on first use per record width against MPFR. MPFR's exponent range is per thread, so each worker thread sets it before it computes.

## 4. tanh

$\tanh$ is computed in the format from the five operations:

- $\tanh|z| = -E/(E+2)$ with $E = \mathrm{expm1}(-2|z|)$.
- $\mathrm{expm1}$ uses Cody–Waite reduction $u = k\ln 2 + r$, with $\ln 2$ split into a short high part (so $k\ln 2_{hi}$ is exact) and a low part.
- A Horner Taylor polynomial of degree $d(p)$, the smallest with truncation below $2^{-(p+1)}$.
- Reconstruction by exact scaling with $2^k$.
- Saturation to $\pm1$ past $(p+3)\ln 2/2$.

The result is within 3 ulp at every $p$ tested, from 8 to 53 and at 64, 113 and 300 bits (against MPFR). It is not correctly rounded, and it is not the platform's `tanh`: a correctly rounded tanh needs extra internal precision, which the model rules out.

## 5. The solvers

Each solver is reference LAPACK 3.12.1, ported statement by statement from the netlib Fortran into C. Every floating-point operation of the Fortran becomes one rounding macro, in Fortran's evaluation order: left to right, parentheses respected, no contraction.

**Routines ported:**

| solver | LAPACK | BLAS |
|---|---|---|
| `lstsq` | `DGELSS` (all paths), `DGEQR2`, `DGELQ2`, `DLARFG`, `DLARF1F`, `DLAPY2`, `DORM2R`, `DORML2`, `DGEBD2` (upper and lower), `DORGL2`/`DORGBR`, `DBDSQR` with `DLARTG`, `DLAS2`, `DLASV2` and `DLASR`, `DRSCL`, `DLASCL` | `DGEMV`, `DGEMM`, `DNRM2` (Blue's algorithm), `DSCAL`, `DCOPY`, `DROT`, `DSWAP` |
| `solve`, `lu_factor`, `lu_solve` | `DGESV`, `DGETRF` → `DGETRF2`, `DGETRS`, `DLASWP` | `IDAMAX`, `DSCAL`, `DTRSM`, `DGEMM` |
| `cholesky`, `cho_factor`, `cho_solve` | `DPOTRF` → `DPOTRF2`, `DPOTRS` | `DTRSM`, `DSYRK` |
| `norm` | | `DNRM2` |

**Details:**
- **Configuration.** The block size is 1: `ILAENV` returns 1.
  - `DGELSS` then runs the unblocked level-2 kernels.
  - `DGETRF` and `DPOTRF` call their recursive versions, `DGETRF2` and `DPOTRF2`, on the whole matrix.

  Blocking reorders the same operations; the unblocked order is the canonical one.
- **Machine parameters.** They are those of the format, derived exactly as Fortran's intrinsics would derive them for a real kind with `DIGITS = p`, `MINEXPONENT = emin + 1` and `MAXEXPONENT = emax + 1`: `DLAMCH`, `LA_CONSTANTS`, the `DNRM2` thresholds, and `EPS**(-0.125)` in `DBDSQR`. `DGETRF2` uses `DLAMCH('S')` to decide between scaling by a reciprocal and dividing.
- **Workspace.** `DGELSS` chooses between two algorithms for $M<N$ by comparing its workspace with a threshold. The port reproduces the optimal workspace a query returns in this configuration.
- **`DGELSS` paths covered:**
  - $M\ge N$, with the QR step when $M\ge 1.6N$ and without it otherwise;
  - $M<N$, through the LQ step when $N \ge 1.6M$ and the workspace suffices, and through direct lower bidiagonalization otherwise;
  - several right-hand sides;
  - `DLASCL` rescaling of $A$ and $B$;
  - $A = 0$.
- **Where fusion is used.** `DLARF1F` applies a Householder reflector as `DGEMV`, two `DAXPY`s and `DGER`. Each of those treats the columns (left side) or rows (right side) independently, so the port runs the four steps per column or row block back to back. Every value receives exactly the reference operations, in the reference order.

**In narrow formats.** LAPACK's safe-scaling logic (the `DNRM2` thresholds, `DLASCL`) was designed for exponent ranges much wider than the precision. In a format like FP16, where the range is comparable to $p$, those branches are taken often, and squares scaled by `DNRM2` can fall into the subnormal range. That is what LAPACK's algorithm does in such a format. To isolate the effect of significand width, use `Format(p)`, whose exponent range is wide. A format whose range is too narrow to hold LAPACK's scaling constants at all raises instead of solving.

## 6. Threads

Loops over independent items run on several threads:
- the columns of the reflector applications;
- the rows or outputs of matrix-vector products;
- the columns of the rotation sweeps in `DBDSQR`;
- the columns of `DGEMM`, `DSYRK` and left-side `DTRSM`, and the rows of right-side `DTRSM`;
- elements, rows and matrix-product rows in the array kernels.

Each item keeps its own operation sequence, so the results are bit-identical for any thread count (tested). Blocks are handed out dynamically, which balances performance and efficiency cores. The format settings of both backends are process-global, so their entry points take a lock.

## 7. Verification

| Claim | How it is checked |
|---|---|
| Each $+,-,\times,\div,\sqrt{\ }$ is correctly rounded | Against MPFR (gmpy2, IEEE subnormals) for every $p$ = 2..53 and the presets. Includes midpoint-constructed cases, the subnormal range, near-overflow values, results near $2^{-1000}$, and a known double-rounding trap. Above 53 bits: $p$ = 54, 64, 113, 200, 237 and 1000, a narrow range at 64 bits, and subnormals and overflow in a 70-bit format with $e_{\max} = 20$ |
| The emulated IEEE formats are the hardware formats | FP32 and FP64 against numpy's float32/float64, and FP16 against numpy's float16 (a rounded float32 result, exact for $p = 11$), with subnormals and overflow. Also against the native builds |
| `lstsq` is reference LAPACK | Bit-identical to netlib `SGELSS`/`DGELSS`, compiled from the vendored source with gfortran and `-ffp-contract=off`, block size 1. Covers 11 shapes across every path; random, rank-deficient and graded matrices; scales that trigger `DLASCL`; three cutoffs; the emulator and the native builds. Also covers the full-size $4801\times1025$ problem it was built for |
| `solve`, `cholesky` and `norm` are reference LAPACK | Bit-identical to netlib `xGESV`, `xGETRF`, `xPOTRF` (both triangles, with garbage in the unread one), `xPOTRS` and `xNRM2`, on the same build, in single and double precision, on both backends. Includes rectangular `xGETRF`, pivots below `DLAMCH('S')` (the division branch of `DGETRF2`), singular and indefinite matrices (the same `INFO`), and `xNRM2` across its three accumulators |
| The backends agree | The MPFR build, run at $p$ = 11, 24, 40, 53, gives the emulator's results bit for bit through `DGELSS`, `DGETRF`/`DGETRS`, `DPOTRF`/`DPOTRS` (both triangles) and `DNRM2` |
| Accuracy at other $p$ | Normwise backward errors of `solve` and `cho_solve` within $4nu$ for $p$ = 8..53, computed exactly; binary128 and 300-bit solves, Cholesky and norms within their error bounds |
| LAPACK's machine constants | Equal to what gfortran computes from the reference source, including libm's `pow` for `EPS**(-1/8)` |
| tanh | Within 3 ulp for $p$ = 8..53, the presets, and 64, 113 and 300 bits. The native builds equal the emulator |
| Nothing computes above $p$ bits | A static audit (`python -m pfloat.audit`) walks clang's syntax tree of the emulator build, with every macro expanded and every branch included. Outside the rounding layer it finds no floating $+,-,\times,\div$, no unrounded conversion, no math call other than exact ones, and no raw literal. The one documented exception is LAPACK's integer crossover `MNTHR = INT(REAL(MIN(M,N))*1.6)`, computed in single precision as reference LAPACK does; it only chooses whether to factor first. Negative controls confirm the audit catches a planted multiply, division, `sqrt` call and unrounded conversion, in both LAPACK ports. The MPFR build compiles the same C with every operation mapped to an MPFR call |
| Thread count doesn't matter | Bit-identical results for 1, 3 and 8 threads (`lstsq`), 1 and 6 threads (`solve`, `cholesky`), 1 and 4 threads (binary128 `lstsq`) |

## 8. Not emulated

- Rounding modes other than nearest-even, and IEEE exception flags. `pfloat.events()` reports counts of non-finite, overflowing and underflowing results from the $p \le 53$ emulator.
- Non-IEEE formats such as OCP FP8 E4M3 (no infinities, a different maximum).
- Blocked LAPACK, and vendor BLAS/LAPACK (OpenBLAS, MKL, Accelerate). These reorder operations, so they don't give the same bits even in binary64.
- Elementary functions other than sqrt and tanh. They raise instead of computing in binary64.
