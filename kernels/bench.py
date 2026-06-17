"""
Stage 2 throughput / scaling benchmark (kernels/DESIGN.md §9.5, §6).

Two things to show:
  1. The fused recurrence kernel vs the torch integer reference (raw speedup).
  2. The in-SRAM Philox payoff: with noise regenerated on-chip the per-member ternary
     transition is NEVER stored in HBM, so population P scales as a pure time knob --
     weight memory is O(V²) (the shared master) instead of O(P·V²). We push P up and
     watch the HBM-noise path's weight memory blow up while the Philox path's stays flat.

    python kernels/bench.py
"""

import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, _HERE)

from spiking_m2rnn.int_membrane import int_recurrence_reference   # noqa: E402
from triton_recurrence import triton_int_recurrence              # noqa: E402
from triton_philox import triton_recurrence_philox               # noqa: E402

DEV = torch.device("cuda")


def _timeit(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def _inputs(P, B, T, K, V, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    kt = torch.randint(0, 2, (P, B, T, K), generator=g, device=DEV, dtype=torch.int32)
    vt = torch.randint(0, 2, (P, B, T, V), generator=g, device=DEV, dtype=torch.int32)
    qt = torch.randint(0, 2, (P, B, T, K), generator=g, device=DEV, dtype=torch.int32)
    Wq = torch.randint(-1, 2, (P, V, V), generator=g, device=DEV, dtype=torch.int32)
    s = torch.randint(0, 4, (P, B, T), generator=g, device=DEV, dtype=torch.int32)
    master = torch.randn(V, V, generator=g, device=DEV, dtype=torch.float32) * 0.1
    return kt, vt, qt, Wq, s, master


def bench_speedup():
    print("== recurrence speedup (torch reference vs fused kernel), P512 B256 T16 V32 ==")
    kt, vt, qt, Wq, s, _ = _inputs(512, 256, 16, 32, 32)
    t_ref = _timeit(lambda: int_recurrence_reference(kt, vt, qt, Wq.float(), s, 2), n=10, warm=3)
    t_ker = _timeit(lambda: triton_int_recurrence(kt, vt, qt, Wq, s, 2))
    print(f"  torch-ref {t_ref:8.1f} ms | kernel {t_ker:6.2f} ms | speedup {t_ref / t_ker:.0f}x\n")


def bench_pscaling():
    print("== population scaling: HBM-noise (per-member Wq in HBM) vs in-SRAM Philox ==")
    print("   B=8 T=16 K=V=32; 'Wq HBM' = per-member transition bytes the Philox path avoids")
    print(f"   {'P':>7} | {'HBM-Wq kernel':>14} | {'Philox kernel':>14} | {'Wq HBM (MB)':>11}")
    for P in [512, 2048, 8192, 32768]:
        kt, vt, qt, Wq, s, master = _inputs(P, 8, 16, 32, 32)
        Wq_i8 = Wq.to(torch.int8)
        t_hbm = _timeit(lambda: triton_int_recurrence(kt, vt, qt, Wq_i8, s, 2), n=10, warm=3)
        t_phx = _timeit(lambda: triton_recurrence_philox(kt, vt, qt, master, s, 2, seed=1),
                        n=10, warm=3)
        wq_mb = P * 32 * 32 / 1e6   # int8 per-member transition that Philox keeps off HBM
        print(f"   {P:>7} | {t_hbm:11.2f} ms | {t_phx:11.2f} ms | {wq_mb:11.1f}")
    print("\n   (Philox weight-memory is O(V²)=4 KB regardless of P; the HBM-Wq column is the\n"
          "    per-member transition that would otherwise be materialized + streamed.)")


if __name__ == "__main__":
    bench_speedup()
    bench_pscaling()
