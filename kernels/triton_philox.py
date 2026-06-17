"""
Stage 2.2 -- in-SRAM Philox noise regeneration (kernels/DESIGN.md §6, the hyperscale
trick). Instead of streaming per-member low-rank factors (A_p, B_p) from HBM, the
recurrence kernel REGENERATES them on-chip from a counter-based RNG keyed by the
integer seed, builds the per-member ternary transition `Wq = ternary(W_master + σ·A B^T)`
in SRAM, uses it, and discards it. Population P then decouples from HBM capacity AND
bandwidth -> a pure time knob.

Invariants enforced (DESIGN §6 / CLAUDE.md guardrail #5):
 1. forward & update regenerate bit-identical noise: both call the SAME `_gen_ab` Philox
    helper with the same (seed, param_off, p) -> identical by construction.
 2. Gaussian factors: `tl.randn` (Philox uniform -> normal), not raw uniform.
 3. antithetic pairing baked into the seed map: member 2m and 2m+1 share (A,B) but A's
    sign flips (±A) -> variance reduction.
 4. Philox (tl.randn), distinct counter offsets for A vs B (A: [0,V), B: [V,2V)) and via
    `param_off` for different matrices/steps -> decorrelated.
 5. Triton-Philox ≠ torch.randn: validation regenerates with the SAME Philox by dumping
    the kernel's own (A,B)/Wq (see kernels/test_philox.py), never torch.randn.

rank=1 (RANK_SCALE=1): W_eff[o,i] = W_master[o,i] + σ·A[o]·B[i].
"""

import torch
import triton
import triton.language as tl

from triton_recurrence import _recur_body

_rint = tl.extra.cuda.libdevice.rint                 # round half-to-even == torch.round


@triton.jit
def _gen_ab(seed, p, V, BV: tl.constexpr, param_off, ANTITHETIC: tl.constexpr):
    """Regenerate rank-1 factors A (over out o) and B (over in i) for member p.
    Returns (A, B) as [BV] float32 (valid for index < V)."""
    rv = tl.arange(0, BV)
    if ANTITHETIC:
        pair = p >> 1
        sign = 1.0 - 2.0 * (p & 1).to(tl.float32)    # +1 even, -1 odd
    else:
        pair = p
        sign = 1.0
    base = param_off + pair * (2 * V)
    A = tl.randn(seed, base + rv) * sign             # counter offsets [0,V)
    B = tl.randn(seed, base + V + rv)                # counter offsets [V,2V)
    return A, B


@triton.jit
def _build_wq(seed, p, V, BV: tl.constexpr, param_off, sigma_rs, master_ptr,
             ANTITHETIC: tl.constexpr):
    """Build the per-member ternary transition Wq[o,i] ∈ {-1,0,1} in SRAM (BitNet absmean),
    returning both Wq[o,i] and its transpose WqT[i,o] (int8, for tl.dot)."""
    rv = tl.arange(0, BV)
    mv = rv < V
    A, B = _gen_ab(seed, p, V, BV, param_off, ANTITHETIC)
    mask2 = mv[:, None] & mv[None, :]
    master = tl.load(master_ptr + rv[:, None] * V + rv[None, :], mask=mask2, other=0.0)
    W_eff = master + sigma_rs * (A[:, None] * B[None, :])             # [o,i]
    absmean = tl.sum(tl.where(mask2, tl.abs(W_eff), 0.0)) / (V * V)
    absmean = tl.maximum(absmean, 1e-5)
    Wq = tl.minimum(tl.maximum(_rint(W_eff / absmean), -1.0), 1.0)    # clamp to {-1,0,1}
    Wq = tl.where(mask2, Wq, 0.0)
    return Wq, tl.trans(Wq).to(tl.int8)                              # [o,i], [i,o]


@triton.jit
def _recurrence_philox_kernel(
    kt_ptr, vt_ptr, qt_ptr, master_ptr, s_ptr, y_ptr, fire_ptr, wq_dump_ptr,
    B, T, K, V, theta, outer_gain, seed, param_off, sigma_rs,
    BK: tl.constexpr, BV: tl.constexpr, ANTITHETIC: tl.constexpr, DUMP_WQ: tl.constexpr,
):
    pid = tl.program_id(0)
    p = pid // B
    b = pid % B
    Wq, wqT = _build_wq(seed, p, V, BV, param_off, sigma_rs, master_ptr, ANTITHETIC)
    if DUMP_WQ:
        if b == 0:                                   # Wq is shared across b -> dump once per p
            rv = tl.arange(0, BV)
            mv = rv < V
            tl.store(wq_dump_ptr + p * V * V + rv[:, None] * V + rv[None, :],
                     Wq, mask=mv[:, None] & mv[None, :])
    fire = _recur_body(kt_ptr, qt_ptr, vt_ptr, s_ptr, y_ptr, wqT, pid,
                       T, K, V, theta, outer_gain, BK, BV)
    tl.store(fire_ptr + pid, fire)


@triton.jit
def _gen_ab_kernel(a_ptr, b_ptr, V, param_off, seed,
                   BV: tl.constexpr, ANTITHETIC: tl.constexpr):
    """Dump regenerated (A,B) to HBM for the update kernel / validation (invariant #1)."""
    p = tl.program_id(0)
    rv = tl.arange(0, BV)
    mv = rv < V
    A, Bv = _gen_ab(seed, p, V, BV, param_off, ANTITHETIC)
    tl.store(a_ptr + p * V + rv, A, mask=mv)
    tl.store(b_ptr + p * V + rv, Bv, mask=mv)


def triton_recurrence_philox(kt, vt, qt, master, s, theta, seed, param_off=0,
                             sigma=0.05, rank_scale=1.0, antithetic=True,
                             return_fire=False, dump_wq=False, outer_gain=1):
    """Recurrence with the transition noise regenerated in-SRAM via Philox.

    master (V,V) float W-master (shared across members); seed/param_off key the Philox
    counter. Returns Y (P,B,T,V) int32 (+ firing if asked; + dumped Wq (P,V,V) if dump_wq).
    """
    P, Bn, T, K = kt.shape
    V = vt.shape[-1]
    dev = kt.device
    kt_i = kt.to(torch.int8).contiguous()
    vt_i = vt.to(torch.int8).contiguous()
    qt_i = qt.to(torch.int8).contiguous()
    s_i = s.to(torch.int32).contiguous()
    master_f = master.to(torch.float32).contiguous()
    Y = torch.empty(P, Bn, T, V, dtype=torch.int32, device=dev)
    fire = torch.empty(P * Bn, dtype=torch.int32, device=dev)
    wq_dump = torch.zeros(P, V, V, dtype=torch.float32, device=dev) if dump_wq else \
        torch.empty(1, dtype=torch.float32, device=dev)
    BK = max(32, triton.next_power_of_2(K))
    BV = max(32, triton.next_power_of_2(V))
    _recurrence_philox_kernel[(P * Bn,)](
        kt_i, vt_i, qt_i, master_f, s_i, Y, fire, wq_dump,
        Bn, T, K, V, int(theta), int(outer_gain), int(seed), int(param_off),
        float(sigma * rank_scale),
        BK=BK, BV=BV, ANTITHETIC=antithetic, DUMP_WQ=dump_wq,
    )
    out = (Y,)
    if return_fire:
        out = out + (fire.sum().item() / (P * Bn * T * K * V),)
    if dump_wq:
        out = out + (wq_dump,)
    return out[0] if len(out) == 1 else out


def gen_noise(P, V, seed, param_off=0, antithetic=True, device="cuda"):
    """Regenerate (A,B) factors (P,V) via the SAME Philox the recurrence kernel uses
    (invariant #1) -- for the ES update and for validation."""
    BV = max(32, triton.next_power_of_2(V))
    A = torch.empty(P, V, dtype=torch.float32, device=device)
    Bf = torch.empty(P, V, dtype=torch.float32, device=device)
    _gen_ab_kernel[(P,)](A, Bf, V, int(param_off), int(seed), BV=BV, ANTITHETIC=antithetic)
    return A, Bf
