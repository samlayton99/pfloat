"""Timing of pbit.lstsq against the problem size, the format and the thread count.

    python benchmarks/bench.py            # prints a markdown table
"""
from __future__ import annotations

import os
import platform
import time

import numpy as np

import pbit


def timed(fn, repeat=2):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    rng = np.random.default_rng(0)
    threads = os.cpu_count() or 1
    rows = []
    for m, n in ((1000, 200), (2401, 513), (4801, 1025)):
        a0, b0 = rng.standard_normal((m, n)), rng.standard_normal(m)
        t_np = timed(lambda: np.linalg.lstsq(a0, b0, rcond=None))
        for label, fmt, backend in (("p = 24 (emulated)", pbit.Format(24), "emulator"),
                                    ("p = 40 (emulated)", pbit.Format(40), "emulator"),
                                    ("fp32 (native path)", pbit.FP32, "auto"),
                                    ("fp64 (native path)", pbit.FP64, "auto")):
            a, b = pbit.array(a0, fmt), pbit.array(b0, fmt)
            times = []
            for th in (1, threads):
                pbit.set_num_threads(th)
                times.append(timed(lambda: pbit.linalg._solve(a, b, [fmt.eps * max(m, n)], backend),
                                   repeat=1 if m > 3000 else 2))
            rows.append((f"{m} x {n}", label, times[0], times[1], t_np))
            print(rows[-1], flush=True)
    print(f"\n{platform.machine()} / {platform.system()}, {threads} threads; numpy {np.__version__}\n")
    print(f"| problem | format | 1 thread | {threads} threads | numpy float64 lstsq |")
    print("|---|---|---:|---:|---:|")
    for size, label, t1, tn, tnp in rows:
        print(f"| {size} | {label} | {t1:.2f} s | {tn:.2f} s | {tnp:.3f} s |")


if __name__ == "__main__":
    main()
