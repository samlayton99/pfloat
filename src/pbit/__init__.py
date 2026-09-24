"""pbit: p-bit floating point, a correctly rounded array type, and a LAPACK least-squares solver.

Choose any significand width p from 2 to 53 bits (and an exponent range). Every +, -, *, /, sqrt
is then correctly rounded to that format (IEEE 754 round to nearest, ties to even, gradual
underflow), nothing is computed in higher precision behind your back, and ``pbit.lstsq`` solves
least squares with reference LAPACK's DGELSS ported operation by operation, bit-identical to
netlib SGELSS/DGELSS in binary32/binary64.

    >>> import pbit
    >>> A = pbit.array([[1, 2], [3, 4], [5, 6]], fmt=16)      # a 16-bit significand
    >>> x, residuals, rank, s = pbit.lstsq(A, [1, 2, 2])
    >>> x.to_numpy()                                          # exact binary64 copy of the p-bit values
"""
from ._formats import BF16, FP8_E5M2, FP16, FP32, FP64, TF32, Format, as_format, round_rational
from ._array import (PArray, absolute, array, asarray, concatenate, dot, events, full, full_like,
                     get_format, matmul, maximum, minimum, ones, ones_like, precision, round_to,
                     set_format, sqrt, stack, sum, tanh, where, zeros, zeros_like)
from . import linalg
from ._lib import get_num_threads, set_num_threads
from .linalg import LinAlgError, lstsq, lstsq_cutoffs

__version__ = "0.1.0"

__all__ = [
    "Format", "FP64", "FP32", "FP16", "BF16", "TF32", "FP8_E5M2", "as_format", "round_rational",
    "PArray", "array", "asarray", "zeros", "ones", "full", "zeros_like", "ones_like", "full_like",
    "precision", "get_format", "set_format", "events", "round_to",
    "sqrt", "tanh", "sum", "dot", "matmul", "maximum", "minimum", "absolute", "where",
    "concatenate", "stack", "linalg", "lstsq", "lstsq_cutoffs", "LinAlgError", "set_num_threads",
    "get_num_threads", "__version__",
]
