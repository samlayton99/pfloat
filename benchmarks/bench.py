"""Timing of pfloat's solvers against the problem size, the format and the thread count.

    python benchmarks/bench.py            # prints two markdown tables: lstsq, then solve / cholesky
"""
from __future__ import annotations

import os
import platform
import time

import numpy as np

import pfloat

FORMATS = (("p = 24 (emulated)", pfloat.Format(24), "emulator"),
           ("p = 40 (emulated)", pfloat.Format(40), "emulator"),
           ("fp32 (native path)", pfloat.FP32, "auto"),
           ("fp64 (native path)", pfloat.FP64, "auto"),
           ("fp128 (MPFR)", pfloat.FP128, "auto"),
           ("fp256 (MPFR)", pfloat.FP256, "auto"))


def timed(fn, repeat=2):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def both_thread_counts(fn, threads, repeat):
    out = []
    for th in (1, threads):
        pfloat.set_num_threads(th)
        out.append(timed(fn, repeat))
    return out


def main():
    rng = np.random.default_rng(0)
    threads = os.cpu_count() or 1
    lstsq_rows, square_rows = [], []
    for m, n in ((1000, 200), (2401, 513), (4801, 1025)):
        a0, b0 = rng.standard_normal((m, n)), rng.standard_normal(m)
        t_np = timed(lambda: np.linalg.lstsq(a0, b0, rcond=None))
        for label, fmt, backend in FORMATS:
            if fmt.extended and m > 3000:
                continue
            a, b = pfloat.array(a0, fmt), pfloat.array(b0, fmt)
            t1, tn = both_thread_counts(lambda: pfloat.lstsq(a, b, backend=backend), threads, 1 if m > 2000 else 2)
            lstsq_rows.append((f"{m} x {n}", label, t1, tn, t_np))
            print(lstsq_rows[-1], flush=True)
    for n in (500, 1000):
        a0, b0 = rng.standard_normal((n, n)), rng.standard_normal(n)
        s0 = a0.T @ a0 + n * np.eye(n)
        t_np = (timed(lambda: np.linalg.solve(a0, b0)), timed(lambda: np.linalg.cholesky(s0)))
        for label, fmt, backend in FORMATS:
            a, b, s = pfloat.array(a0, fmt), pfloat.array(b0, fmt), pfloat.array(s0, fmt)
            ts = both_thread_counts(lambda: pfloat.solve(a, b, backend=backend), threads, 1)
            tc = both_thread_counts(lambda: pfloat.cholesky(s, backend=backend), threads, 1)
            square_rows.append((f"{n} x {n}", label, *ts, *tc, *t_np))
            print(square_rows[-1], flush=True)
    print(f"\n{platform.machine()} / {platform.system()}, {threads} threads; numpy {np.__version__}\n")
    print(f"| problem | format | 1 thread | {threads} threads | numpy float64 lstsq |")
    print("|---|---|---:|---:|---:|")
    for size, label, t1, tn, tnp in lstsq_rows:
        print(f"| {size} | {label} | {t1:.2f} s | {tn:.2f} s | {tnp:.3f} s |")
    print(f"\n| matrix | format | solve, 1 thread | solve, {threads} threads | cholesky, 1 thread | "
          f"cholesky, {threads} threads |")
    print("|---|---|---:|---:|---:|---:|")
    for size, label, s1, sn, c1, cn, snp, cnp in square_rows:
        print(f"| {size} | {label} | {s1:.2f} s | {sn:.2f} s | {c1:.2f} s | {cn:.2f} s |")
    for size, *_, snp, cnp in square_rows[::len(FORMATS)]:
        print(f"numpy float64 at {size}: solve {snp:.4f} s, cholesky {cnp:.4f} s")


if __name__ == "__main__":
    main()
