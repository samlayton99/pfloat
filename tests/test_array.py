"""The PArray data type: construction, conversion, arithmetic, and refusal of silent binary64."""
import decimal
from fractions import Fraction

import numpy as np
import pytest

import pfloat
from conftest import mpfr_context


def test_construction_rounds_correctly_from_every_source():
    import gmpy2
    import mpmath
    fmt = pfloat.Format(10)
    third = Fraction(1, 3)
    with mpfr_context(fmt):
        want = float(gmpy2.mpfr(1) / 3)
    assert float(pfloat.array(third, fmt)) == want
    assert float(pfloat.array(decimal.Decimal(1) / decimal.Decimal(3), fmt)) == want
    with mpmath.workprec(200):
        assert float(pfloat.array(mpmath.mpf(1) / 3, fmt)) == want
    with gmpy2.context(precision=200):
        assert float(pfloat.array(gmpy2.mpfr(1) / 3, fmt)) == want
    big = 2 ** 70 + 2 ** 60 + 1
    with mpfr_context(fmt):
        assert float(pfloat.array(big, fmt)) == float(gmpy2.mpfr(big))
    for dtype in (np.float16, np.float32, np.float64, np.int32, np.int64, np.uint8):
        x = np.arange(-5, 6).astype(dtype) if dtype != np.uint8 else np.arange(11).astype(dtype)
        a = pfloat.array(x, fmt)
        np.testing.assert_array_equal(a.to_numpy(), pfloat.round_to(x.astype(np.float64), fmt))
    assert pfloat.array([[1, 2], [3, 4]], fmt).shape == (2, 2)


def test_torch_roundtrip():
    torch = pytest.importorskip("torch")
    t = torch.linspace(-3, 3, 101, dtype=torch.float32)
    a = pfloat.array(t, pfloat.FP32)
    assert torch.equal(a.to_torch(torch.float32), t)
    b = pfloat.array(t, 8)
    assert b.to_torch().dtype == torch.float64
    assert torch.equal(b.to_torch(torch.bfloat16).double(), b.to_torch())  # 8-bit values fit bf16 exactly
    with pytest.raises(ValueError):
        pfloat.array(torch.tensor([1 / 3], dtype=torch.float64), pfloat.FP64).to_torch(torch.float16)


def test_conversions_out_are_exact_or_refused():
    a = pfloat.array([0.1, 0.2, 1 / 3], 24)
    assert a.to_numpy(np.float32).dtype == np.float32
    np.testing.assert_array_equal(a.to_numpy(np.float32).astype(np.float64), a.to_numpy())
    with pytest.raises(ValueError):
        pfloat.array([0.1], 40).to_numpy(np.float32)
    assert np.asarray(a).dtype == np.float64
    assert a.to_fractions()[0] == Fraction(float(a[0]))
    assert a.tolist() == a.to_numpy().tolist()


def test_arithmetic_is_elementwise_correct_rounding_with_broadcasting():
    import gmpy2
    fmt = pfloat.Format(9)
    rng = np.random.default_rng(0)
    x = pfloat.array(rng.standard_normal((4, 5)), fmt)
    y = pfloat.array(rng.standard_normal(5), fmt)
    for op, fn in ((np.add, lambda a, b: a + b), (np.subtract, lambda a, b: a - b),
                   (np.multiply, lambda a, b: a * b), (np.divide, lambda a, b: a / b)):
        got = fn(x, y).to_numpy()
        with mpfr_context(fmt):
            want = np.vectorize(lambda a, b: float(fn(gmpy2.mpfr(a), gmpy2.mpfr(b))))(*np.broadcast_arrays(x.to_numpy(), y.to_numpy()))
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(op(x, y).to_numpy(), want)  # the numpy ufunc routes to pfloat
    # plain numbers are rounded into the format first
    np.testing.assert_array_equal((x + 0.1).to_numpy(), (x + pfloat.array(0.1, fmt)).to_numpy())
    np.testing.assert_array_equal((1 - x).to_numpy(), (pfloat.array(1, fmt) - x).to_numpy())


def test_sum_and_matmul_orders():
    fmt = pfloat.Format(6)
    v = pfloat.array([1.0, 2 ** -6, 2 ** -6, 2 ** -6, 2 ** -6], fmt)
    s = pfloat.array(0, fmt)
    s = v[0]
    for t in v[1:]:
        s = s + t
    assert float(pfloat.sum(v)) == float(s) == float(np.sum(v))  # left to right: the small terms vanish
    rng = np.random.default_rng(1)
    A, B = pfloat.array(rng.standard_normal((3, 7)), fmt), pfloat.array(rng.standard_normal((7, 2)), fmt)
    C = A @ B
    for i in range(3):
        for j in range(2):
            acc = pfloat.array(0, fmt)
            for k in range(7):
                acc = acc + A[i, k] * B[k, j]
            assert float(C[i, j]) == float(acc)
    assert (A @ B[:, 0]).shape == (3,) and pfloat.dot(A[0], A[1]).shape == ()


def test_unimplemented_numpy_raises_instead_of_leaking():
    x = pfloat.array([1.0, 2.0], 20)
    for bad in (np.exp, np.log, np.sin):
        with pytest.raises(TypeError, match="not implemented at p-bit precision"):
            bad(x)
    with pytest.raises(TypeError, match="not implemented at p-bit precision"):
        np.mean(x)
    with pytest.raises(TypeError, match="not implemented at p-bit precision"):
        np.cumsum(x)
    with pytest.raises(TypeError):
        x ** 3


def test_mixed_formats_and_missing_format_raise():
    with pytest.raises(TypeError, match="mixed formats"):
        pfloat.array([1.0], 10) + pfloat.array([1.0], 11)
    with pytest.raises(TypeError, match="no format given"):
        pfloat.array([1.0])


def test_precision_context_and_astype():
    with pfloat.precision(pfloat.BF16):
        a = pfloat.array([1 / 3])
        assert a.fmt == pfloat.BF16
        with pfloat.precision(12):
            assert pfloat.array([1.0]).fmt == pfloat.Format(12)
        assert pfloat.get_format() == pfloat.BF16
    assert pfloat.get_format() is None
    b = pfloat.array([1 / 3], 40).astype(pfloat.BF16)
    np.testing.assert_array_equal(b.to_numpy(), pfloat.round_to([1 / 3], pfloat.BF16))


def test_exact_operations_and_views():
    fmt = pfloat.Format(12)
    x = pfloat.array([[-1.5, 2.0], [3.0, -0.25]], fmt)
    assert np.abs(x).fmt == fmt and float(np.max(np.abs(x).to_numpy())) == 3.0
    assert x.T.shape == (2, 2) and x.reshape(4).shape == (4,)
    assert isinstance(x[0], pfloat.PArray) and float(x[1, 0]) == 3.0
    x[0, 0] = 1 / 3
    assert float(x[0, 0]) == float(pfloat.array(1 / 3, fmt))
    assert (x > 0).dtype == bool
    assert pfloat.concatenate([x, x]).shape == (4, 2)
    assert np.sqrt(pfloat.array([2.0], fmt)).fmt == fmt
    assert np.tanh(pfloat.array([0.5], fmt)).fmt == fmt


def test_formats():
    assert pfloat.FP32.max == float(np.finfo(np.float32).max)
    assert pfloat.FP16.min_subnormal == 2.0 ** -24 and pfloat.BF16.eps == 2.0 ** -7
    assert pfloat.as_format("bf16") == pfloat.BF16 and pfloat.as_format(20) == pfloat.Format(20)
    assert pfloat.Format(54).extended and not pfloat.Format(53).extended
    with pytest.raises(ValueError):
        pfloat.Format(4097)
    with pytest.raises(ValueError):
        pfloat.Format(40, emin=-1000, emax=1000)
