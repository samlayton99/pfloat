"""Least-squares error against working precision: one bit of error per bit of precision.

Fits f(x) = exp(sin(3x)) with 40 Chebyshev polynomials on 400 points, with every step at p bits:
the data are rounded to p bits, the Chebyshev features are built by the three-term recurrence in
p-bit arithmetic, the solve is pfloat.lstsq (DGELSS at p bits), and the fit is evaluated at p bits.
The error is measured in binary64 against the exact function.

    python examples/precision_sweep.py       # prints a table; saves precision_sweep.png if matplotlib is installed
"""
from pathlib import Path

import numpy as np

import pfloat


def chebyshev_features(x: pfloat.PArray, degree: int) -> pfloat.PArray:
    cols = [pfloat.ones(x.shape, x.fmt), x]
    for _ in range(degree - 1):
        cols.append(2 * x * cols[-1] - cols[-2])      # T_{k+1} = 2 x T_k - T_{k-1}, at p bits
    return pfloat.stack(cols, axis=1)


def main():
    f = lambda t: np.exp(np.sin(3 * t))                  # noqa: E731
    x_fit, x_eval = np.linspace(-1, 1, 400), np.linspace(-1, 1, 2001)
    rows = []
    for p in range(8, 54):
        fmt = pfloat.Format(p)
        A = chebyshev_features(pfloat.array(x_fit, fmt), 39)
        coef, _, rank, _ = pfloat.lstsq(A, pfloat.array(f(x_fit), fmt))
        fit = chebyshev_features(pfloat.array(x_eval, fmt), 39) @ coef
        err = np.max(np.abs(fit.to_numpy() - f(x_eval)))
        rows.append((p, err, rank))
        print(f"p = {p:2d}   max error = {err:.3e}   error * 2^p = {err * 2.0 ** p:8.1f}   rank = {rank}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ps = np.array([r[0] for r in rows])
    errs = np.array([r[1] for r in rows])
    c = np.exp2(np.median(np.log2(errs[(ps >= 12) & (ps <= 46)]) + ps[(ps >= 12) & (ps <= 46)]))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(ps, errs, "o-", ms=3.5, lw=1.8, color="#2a788e", label="max error, everything at p bits")
    ax.semilogy(ps, c * 2.0 ** -ps, "--", color="#333333", lw=1.2, label=f"{c:.0f} * 2^-p")
    ax.set_xlabel("working precision p (bits)")
    ax.set_ylabel("max |fit - f|")
    ax.set_xlim(8, 53)
    ax.set_ylim(1e-17, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9, borderaxespad=0)
    fig.tight_layout()
    out = Path(__file__).with_name("precision_sweep.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
