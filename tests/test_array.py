"""The PArray data type: construction, conversion, arithmetic, and refusal of silent binary64."""
import decimal
from fractions import Fraction

import numpy as np
import pytest

import pbit
from conftest import mpfr_context


def test_construction_rounds_correctly_from_every_source():
    import gmpy2
    import mpmath
    fmt = pbit.Format(10)
    third = Fraction(1, 3)
    with mpfr_context(fmt):
        want = float(gmpy2.mpfr(1) / 3)
    assert float(pbit.array(third, fmt)) == want
    assert float(pbit.array(decimal.Decimal(1) / decimal.Decimal(3), fmt)) == want
    with mpmath.workprec(200):
        assert float(pbit.array(mpmath.mpf(1) / 3, fmt)) == want
    with gmpy2.context(precision=200):
        assert float(pbit.array(gmpy2.mpfr(1) / 3, fmt)) == want
    big = 2 ** 70 + 2 ** 60 + 1
    with mpfr_context(fmt):
        assert float(pbit.array(big, fmt)) == float(gmpy2.mpfr(big))
    for dtype in (np.float16, np.float32, np.float64, np.int32, np.int64, np.uint8):
        x = np.arange(-5, 6).astype(dtype) if dtype != np.uint8 else np.arange(11).astype(dtype)
        a = pbit.array(x, fmt)
        np.testing.assert_array_equal(a.to_numpy(), pbit.round_to(x.astype(np.float64), fmt))
    assert pbit.array([[1, 2], [3, 4]], fmt).shape == (2, 2)


def test_torch_roundtrip():
    torch = pytest.importorskip("torch")
    t = torch.linspace(-3, 3, 101, dtype=torch.float32)
    a = pbit.array(t, pbit.FP32)
    assert torch.equal(a.to_torch(torch.float32), t)
    b = pbit.array(t, 8)
    assert b.to_torch().dtype == torch.float64
    assert torch.equal(b.to_torch(torch.bfloat16).double(), b.to_torch())  # 8-bit values fit bf16 exactly
    with pytest.raises(ValueError):
        pbit.array(torch.tensor([1 / 3], dtype=torch.float64), pbit.FP64).to_torch(torch.float16)


def test_conversions_out_are_exact_or_refused():
    a = pbit.array([0.1, 0.2, 1 / 3], 24)
    assert a.to_numpy(np.float32).dtype == np.float32
    np.testing.assert_array_equal(a.to_numpy(np.float32).astype(np.float64), a.to_numpy())
    with pytest.raises(ValueError):
        pbit.array([0.1], 40).to_numpy(np.float32)
    assert np.asarray(a).dtype == np.float64
    assert a.to_fractions()[0] == Fraction(float(a[0]))
    assert a.tolist() == a.to_numpy().tolist()


def test_arithmetic_is_elementwise_correct_rounding_with_broadcasting():
    import gmpy2
    fmt = pbit.Format(9)
    rng = np.random.default_rng(0)
    x = pbit.array(rng.standard_normal((4, 5)), fmt)
    y = pbit.array(rng.standard_normal(5), fmt)
    for op, fn in ((np.add, lambda a, b: a + b), (np.subtract, lambda a, b: a - b),
                   (np.multiply, lambda a, b: a * b), (np.divide, lambda a, b: a / b)):
        got = fn(x, y).to_numpy()
        with mpfr_context(fmt):
            want = np.vectorize(lambda a, b: float(fn(gmpy2.mpfr(a), gmpy2.mpfr(b))))(*np.broadcast_arrays(x.to_numpy(), y.to_numpy()))
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(op(x, y).to_numpy(), want)  # the numpy ufunc routes to pbit
    # plain numbers are rounded into the format first
    np.testing.assert_array_equal((x + 0.1).to_numpy(), (x + pbit.array(0.1, fmt)).to_numpy())
    np.testing.assert_array_equal((1 - x).to_numpy(), (pbit.array(1, fmt) - x).to_numpy())


def test_sum_and_matmul_orders():
    fmt = pbit.Format(6)
    v = pbit.array([1.0, 2 ** -6, 2 ** -6, 2 ** -6, 2 ** -6], fmt)
    s = pbit.array(0, fmt)
    s = v[0]
    for t in v[1:]:
        s = s + t
    assert float(pbit.sum(v)) == float(s) == float(np.sum(v))  # left to right: the small terms vanish
    rng = np.random.default_rng(1)
    A, B = pbit.array(rng.standard_normal((3, 7)), fmt), pbit.array(rng.standard_normal((7, 2)), fmt)
    C = A @ B
    for i in range(3):
        for j in range(2):
            acc = pbit.array(0, fmt)
            for k in range(7):
                acc = acc + A[i, k] * B[k, j]
            assert float(C[i, j]) == float(acc)
    assert (A @ B[:, 0]).shape == (3,) and pbit.dot(A[0], A[1]).shape == ()


def test_unimplemented_numpy_raises_instead_of_leaking():
    x = pbit.array([1.0, 2.0], 20)
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
        pbit.array([1.0], 10) + pbit.array([1.0], 11)
    with pytest.raises(TypeError, match="no format given"):
        pbit.array([1.0])


def test_precision_context_and_astype():
    with pbit.precision(pbit.BF16):
        a = pbit.array([1 / 3])
        assert a.fmt == pbit.BF16
        with pbit.precision(12):
            assert pbit.array([1.0]).fmt == pbit.Format(12)
        assert pbit.get_format() == pbit.BF16
    assert pbit.get_format() is None
    b = pbit.array([1 / 3], 40).astype(pbit.BF16)
    np.testing.assert_array_equal(b.to_numpy(), pbit.round_to([1 / 3], pbit.BF16))


def test_exact_operations_and_views():
    fmt = pbit.Format(12)
    x = pbit.array([[-1.5, 2.0], [3.0, -0.25]], fmt)
    assert np.abs(x).fmt == fmt and float(np.max(np.abs(x).to_numpy())) == 3.0
    assert x.T.shape == (2, 2) and x.reshape(4).shape == (4,)
    assert isinstance(x[0], pbit.PArray) and float(x[1, 0]) == 3.0
    x[0, 0] = 1 / 3
    assert float(x[0, 0]) == float(pbit.array(1 / 3, fmt))
    assert (x > 0).dtype == bool
    assert pbit.concatenate([x, x]).shape == (4, 2)
    assert np.sqrt(pbit.array([2.0], fmt)).fmt == fmt
    assert np.tanh(pbit.array([0.5], fmt)).fmt == fmt


def test_formats():
    assert pbit.FP32.max == float(np.finfo(np.float32).max)
    assert pbit.FP16.min_subnormal == 2.0 ** -24 and pbit.BF16.eps == 2.0 ** -7
    assert pbit.as_format("bf16") == pbit.BF16 and pbit.as_format(20) == pbit.Format(20)
    with pytest.raises(ValueError):
        pbit.Format(54)
    with pytest.raises(ValueError):
        pbit.Format(40, emin=-1000, emax=1000)
