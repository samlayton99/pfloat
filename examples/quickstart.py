"""pbit in five minutes. Run: python examples/quickstart.py"""
import numpy as np

import pbit

# 1. Formats: any significand width p from 2 to 53 bits, or an IEEE preset.
f12 = pbit.Format(12)                        # 12-bit significand, wide exponent range
print(f12, "eps =", f12.eps)
print(pbit.FP16, pbit.BF16, pbit.FP32, pbit.FP64)

# 2. Arrays: values are rounded into the format once (nearest, ties to even) ...
x = pbit.array([0.1, 0.2, 0.3], f12)
print(x)                                     # the 12-bit values, printed exactly
# ... and every operation is correctly rounded in the format.
print(x + x * 3, pbit.sqrt(x), pbit.tanh(x))
print(x.sum(), x @ x)                        # left-to-right sum; dot product in BLAS order

# 3. numpy calls that pbit implements run in the format; others raise instead of leaking.
print(np.add(x, 1), np.sqrt(x))
try:
    np.exp(x)
except TypeError as err:
    print("refused:", err)

# 4. Converting out is explicit and exact.
print(x.to_numpy(), x.to_numpy().dtype)      # binary64 copy of the 12-bit values
print(pbit.array([1 / 3], pbit.FP32).to_numpy(np.float32))

# 5. Least squares, with numpy's signature, run entirely in the format.
rng = np.random.default_rng(0)
A = rng.standard_normal((100, 5))
b = A @ np.arange(1.0, 6.0) + 1e-3 * rng.standard_normal(100)
for fmt in (pbit.BF16, pbit.FP16, pbit.Format(16), pbit.FP32, pbit.FP64):
    sol, residuals, rank, s = pbit.lstsq(A, b, fmt=fmt)
    print(f"{fmt!r:>10}: x = {np.round(sol.to_numpy(), 6)}")

# 6. A default format for a block of code.
with pbit.precision(20):
    y = pbit.array(np.linspace(0, 1, 5))
    print(y.fmt, pbit.tanh(y))
