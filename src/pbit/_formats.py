"""Binary floating-point formats with a p-bit significand, and their exact constants."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from math import ceil, factorial, floor, inf

import mpmath
import numpy as np

__all__ = ["Format", "FP64", "FP32", "FP16", "BF16", "TF32", "FP8_E5M2", "as_format", "round_rational"]

DEFAULT_EMIN, DEFAULT_EMAX = -958, 959


@dataclass(frozen=True)
class Format:
    """A binary floating-point format: p significand bits (2..53) and IEEE-style exponents.

    Values are 0, +-inf, NaN and +-m 2^(e-p+1) with integer significand m < 2^p and exponent
    e >= emin (subnormals below 2^emin); normal exponents run emin..emax with emin = 1 - emax, as
    in IEEE 754 (give either one). Arithmetic in the format is IEEE 754 round to nearest, ties to
    even, with gradual underflow and overflow to infinity.

    The default exponent range, [-958, 959], is binary64's pulled in by 64 binades at each end:
    the widest range the emulator represents exactly for every p. Presets: FP64, FP32, FP16,
    BF16, TF32, FP8_E5M2.
    """

    p: int
    emin: int | None = None
    emax: int | None = None
    name: str | None = field(default=None, compare=False)

    def __post_init__(self):
        if not isinstance(self.p, (int, np.integer)) or isinstance(self.p, bool) or not 2 <= self.p <= 53:
            raise ValueError(f"p must be an integer in 2..53, got {self.p!r}")
        emin, emax = self.emin, self.emax
        if emin is None and emax is None:
            emin, emax = DEFAULT_EMIN, DEFAULT_EMAX
        elif emin is None:
            emin = 1 - emax
        elif emax is None:
            emax = 1 - emin
        if emin != 1 - emax:
            raise ValueError(f"IEEE-style exponent range needed: emin = 1 - emax (got emin={emin}, emax={emax})")
        if not 1 <= emax <= 1023:
            raise ValueError("emax must be in 1..1023")
        object.__setattr__(self, "p", int(self.p))
        object.__setattr__(self, "emin", int(emin))
        object.__setattr__(self, "emax", int(emax))
        if not self.is_binary64 and self.emin < -1022 + self.p + 1:
            raise ValueError(f"emin must be >= {-1022 + self.p + 1} for p = {self.p} (the binary64 carrier "
                             "must hold every value of the format, subnormals included), or the format "
                             "must be exactly binary64 (53, -1022, 1023)")

    @property
    def is_binary64(self) -> bool:
        return (self.p, self.emin, self.emax) == (53, -1022, 1023)

    @property
    def eps(self) -> float:
        """Machine epsilon: the spacing of the format at 1, 2^(1-p)."""
        return 2.0 ** (1 - self.p)

    @property
    def unit_roundoff(self) -> float:
        """2^-p: the largest relative error of one rounding (normal range)."""
        return 2.0 ** -self.p

    @property
    def min_normal(self) -> float:
        return 2.0 ** self.emin

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (self.emin - self.p + 1)

    @property
    def max(self) -> float:
        return float((2 - Fraction(2) ** (1 - self.p)) * Fraction(2) ** self.emax)

    def __repr__(self) -> str:
        if self.name:
            return self.name
        if (self.emin, self.emax) == (DEFAULT_EMIN, DEFAULT_EMAX):
            return f"Format(p={self.p})"
        return f"Format(p={self.p}, emin={self.emin}, emax={self.emax})"

    def constants(self) -> np.ndarray:
        """Packed constants for the C kernels (see csrc/kernels.h)."""
        return _packed(self.p, self.emin, self.emax)


FP64 = Format(53, -1022, 1023, "fp64")
FP32 = Format(24, -126, 127, "fp32")
FP16 = Format(11, -14, 15, "fp16")
BF16 = Format(8, -126, 127, "bf16")
TF32 = Format(11, -126, 127, "tf32")
FP8_E5M2 = Format(3, -14, 15, "fp8_e5m2")
PRESETS = {f.name: f for f in (FP64, FP32, FP16, BF16, TF32, FP8_E5M2)}


def as_format(fmt) -> Format:
    """A Format from a Format, an integer p (default exponent range), or a preset name."""
    if isinstance(fmt, Format):
        return fmt
    if isinstance(fmt, (int, np.integer)) and not isinstance(fmt, bool):
        return Format(int(fmt))
    if isinstance(fmt, str) and fmt.lower() in PRESETS:
        return PRESETS[fmt.lower()]
    raise TypeError(f"not a format: {fmt!r} (use a pbit.Format, an integer p, or one of {sorted(PRESETS)})")


# ---------------------------------------------------------------- exact rounding of rationals

def _round_sig(a: Fraction, p: int, emin: int | None = None) -> Fraction:
    """Round a positive rational to p significant bits (nearest, ties to even); below 2^emin
    (when given) the quantum is fixed at 2^(emin - p + 1)."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if emin is not None:
        e = max(e, emin)
    scaled = a * Fraction(2) ** (p - 1 - e)
    m, rem = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * rem
    if twice > scaled.denominator or (twice == scaled.denominator and m % 2):
        m += 1
    return Fraction(m) * Fraction(2) ** (e - p + 1)


def round_rational(value, fmt: Format) -> float:
    """Round an exact rational (int, Fraction, ...) into the format: nearest, ties to even,
    gradual underflow, overflow to infinity. Returns the value as a float (exact)."""
    value = Fraction(value)
    if value == 0:
        return 0.0
    sign = -1 if value < 0 else 1
    out = _round_sig(abs(value), fmt.p, fmt.emin)
    if out > Fraction(fmt.max):
        return sign * inf
    if out == 0:
        return -0.0 if sign < 0 else 0.0
    return float(sign * out)


def _round_real(x, p: int, guard: int = 1500) -> float:
    """Round a real known to 2000 bits to p significant bits; refuse if it is within 2^-guard of
    a rounding midpoint (so the single rounding is certainly correct)."""
    sign, man, exp, _ = mpmath.mpf(x)._mpf_
    a = Fraction(man) * Fraction(2) ** exp
    lo = _round_sig(a * (1 - Fraction(1, 2 ** guard)), p)
    hi = _round_sig(a * (1 + Fraction(1, 2 ** guard)), p)
    assert lo == hi, "constant too close to a rounding midpoint"
    return float(-lo if sign else lo)


@lru_cache(maxsize=None)
def _packed(p: int, emin: int, emax: int) -> np.ndarray:
    """The stored constants of the format, each rounded once into it.

    tanh: 1/ln 2, ln 2 split for Cody-Waite reduction, the saturation point, Taylor coefficients
    1/(j+1)!. LAPACK: what reference LAPACK computes for a Fortran real kind with DIGITS = p,
    MINEXPONENT = emin + 1, MAXEXPONENT = emax + 1 (DLAMCH, LA_CONSTANTS, the DNRM2 thresholds),
    EPS**(-0.125) of DBDSQR as a correctly rounded power, and 0.01.
    """
    fmt = Format(p, emin, emax)
    r = lambda v: round_rational(v, fmt)  # noqa: E731
    with mpmath.workprec(2000):
        ln2 = mpmath.log(2)
        kb = (p + 4).bit_length()
        ln2hi = _round_real(ln2, max(p - kb, 1))
        ln2lo = _round_real(ln2 - mpmath.mpf(ln2hi), p)
        invln2 = _round_real(1 / ln2, p)
        sat = _round_real((p + 3) * ln2 / 2, p)
        epspow = _round_real(mpmath.mpf(2) ** (mpmath.mpf(p) / 8), p)
    d = 1
    while Fraction(35, 100) ** d / factorial(d + 1) > Fraction(1, 2 ** (p + 1)):
        d += 1
    coeffs = [r(Fraction(1, factorial(j + 1))) for j in range(d)]
    two = Fraction(2)
    tiny = two ** emin
    huge = (2 - two ** (1 - p)) * two ** emax
    small = Fraction(r(1 / huge))
    sfmin = tiny
    if small >= tiny:
        sfmin = Fraction(r(small * Fraction(r(1 + two ** -p))))
    safmin_e = max(emin, -emax)
    lapack = [two ** -p, two ** (1 - p), sfmin, huge, two ** safmin_e, two ** -safmin_e,
              two ** ceil(emin * 0.5), two ** floor((emax + 1 - p + 1) * 0.5),
              two ** -floor((emin + 1 - p) * 0.5), two ** -ceil((emax + 1 + p - 1) * 0.5)]
    stored = [r(v) for v in lapack]
    assert stored == [float(v) for v in lapack], f"a LAPACK constant is not a value of {fmt!r}"
    values = [p, emin, emax, r(invln2), r(ln2hi), r(ln2lo), r(sat), *stored,
              r(epspow), r(Fraction(1, 100)), d, *coeffs]
    out = np.array(values, dtype=np.float64)
    out.setflags(write=False)
    return out
