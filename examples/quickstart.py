"""pfloat in five minutes. Run: python examples/quickstart.py"""
import numpy as np

import pfloat

# 1. Formats: any significand width p from 2 to 4096 bits, or an IEEE preset.
f12 = pfloat.Format(12)                        # 12-bit significand, wide exponent range
print(f12, "eps =", f12.eps)
print(pfloat.FP16, pfloat.BF16, pfloat.FP32, pfloat.FP64, pfloat.FP128)

# 2. Arrays: values are rounded into the format once (nearest, ties to even) ...
x = pfloat.array([0.1, 0.2, 0.3], f12)
print(x)                                     # the 12-bit values, printed exactly
# ... and every operation is correctly rounded in the format.
print(x + x * 3, pfloat.sqrt(x), pfloat.tanh(x))
print(x.sum(), x @ x)                        # left-to-right sum; dot product in BLAS order

# 3. numpy calls that pfloat implements run in the format; others raise instead of leaking.
print(np.add(x, 1), np.sqrt(x))
try:
    np.exp(x)
except TypeError as err:
    print("refused:", err)

# 4. Converting out is explicit and exact.
print(x.to_numpy(), x.to_numpy().dtype)      # binary64 copy of the 12-bit values
print(pfloat.array([1 / 3], pfloat.FP32).to_numpy(np.float32))

# 5. Least squares, with numpy's signature, run entirely in the format.
rng = np.random.default_rng(0)
A = rng.standard_normal((100, 5))
b = A @ np.arange(1.0, 6.0) + 1e-3 * rng.standard_normal(100)
for fmt in (pfloat.BF16, pfloat.FP16, pfloat.Format(16), pfloat.FP32, pfloat.FP64):
    sol, residuals, rank, s = pfloat.lstsq(A, b, fmt=fmt)
    print(f"{fmt!r:>10}: x = {np.round(sol.to_numpy(), 6)}")

# 6. Square systems: LU (numpy.linalg.solve) and Cholesky, again entirely in the format.
M = rng.standard_normal((6, 6))
S = M @ M.T + 6 * np.eye(6)
rhs = np.ones(6)
for fmt in (pfloat.Format(16), pfloat.FP64):
    print(f"{fmt!r:>10}: solve {np.round(pfloat.solve(M, rhs, fmt=fmt).to_numpy()[:3], 6)}, "
          f"cholesky diag {np.round(np.diag(pfloat.cholesky(S, fmt=fmt).to_numpy()), 4)[:3]}")

# 7. Above 53 bits (needs MPFR: pip install "pfloat[mp]"), e.g. IEEE binary128.
try:
    sol, *_ = pfloat.lstsq(A, b, fmt=pfloat.FP128)
    print("fp128:", sol[:2], "residual norm", pfloat.norm(A @ sol - b))
except RuntimeError as err:
    print("binary128 unavailable:", err)

# 8. A default format for a block of code.
with pfloat.precision(20):
    y = pfloat.array(np.linspace(0, 1, 5))
    print(y.fmt, pfloat.tanh(y))
