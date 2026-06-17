"""
Stage 2.1 -- fused INTEGER/bitwise ternary-transition recurrence kernel (Triton).

This is the kernel kernels/DESIGN.md specifies: the serial `for t` recurrence of
`SpikingM2RNN` (spike + mac_free + ternary + integer membrane) fused into one Triton
program per (population member p, batch element b), carrying the matrix state `S` and
the integer membrane `mem` in registers/SRAM across time. It is validated **bit-exact
(exact integer equality)** against `spiking_m2rnn.int_membrane.int_recurrence_reference`
(the materialized reference path, guardrail #2/#6).

Op realization (DESIGN §3 dispatch table):
  trans[k,o] = Σ_i state[k,i]·Wq[o,i]   int8 IMMA `tl.dot` (state binary, Wq ternary) -> int32 [-V,V]
  outer[k,o] = k_t[k]·v_t[o]            int8 outer product (broadcast)                -> {0,1}
  mem        = (mem >> s_t)+trans+outer arithmetic right shift (s_t∈{0,1,2,3}) + int add
  S          = mem > θ                  integer compare                                -> {0,1}
  mem        = mem - θ·S                masked subtract (subtractive reset)
  y_t[o]     = Σ_k S[k,o]·q_t[k]        int reduce over k                              -> [0,K]

Parallel axes P·B map to the grid; the contraction/value dims K,V are the in-program
tile (one program owns the whole K×V state so the y-reduction over k needs no atomics).
The transition matrix Wq is per-member (shared across b and t) -- DESIGN §5 reuse.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _recur_body(kt_ptr, qt_ptr, vt_ptr, s_ptr, y_ptr, wqT, pb,
                T, K, V, theta, outer_gain, BK: tl.constexpr, BV: tl.constexpr):
    """The fused per-(p,b) integer recurrence over time, given the ternary transition
    already tiled as wqT[i,o] (int8). Shared by the noise-from-HBM and in-SRAM-Philox
    kernels so the dynamics are byte-identical. Returns the per-program spike count."""
    rk = tl.arange(0, BK)                        # key index   (rows of state)
    rv = tl.arange(0, BV)                        # value index (cols of state / out)
    mk = rk < K
    mv = rv < V
    state = tl.zeros((BK, BV), dtype=tl.int8)    # binary key×value spike matrix S
    mem = tl.zeros((BK, BV), dtype=tl.int32)     # integer membrane
    fire = tl.zeros((), dtype=tl.int32)
    for t in range(T):
        k_off = (pb * T + t) * K
        v_off = (pb * T + t) * V
        kt = tl.load(kt_ptr + k_off + rk, mask=mk, other=0).to(tl.int8)  # [BK] key spikes
        qt = tl.load(qt_ptr + k_off + rk, mask=mk, other=0).to(tl.int32) # [BK] query spikes
        vt = tl.load(vt_ptr + v_off + rv, mask=mv, other=0).to(tl.int8)  # [BV] value spikes
        s_t = tl.load(s_ptr + pb * T + t)                                # scalar shift amount

        outer = (kt[:, None] * vt[None, :]).to(tl.int32) * outer_gain    # [BK,BV] {0,gain}
        trans = tl.dot(state, wqT, out_dtype=tl.int32)                   # [BK,BV] in [-V,V]
        mem = (mem >> s_t) + trans + outer                              # arithmetic shift leak
        S = (mem > theta).to(tl.int32)                                   # [BK,BV] {0,1}
        mem = mem - theta * S                                            # subtractive reset
        y = tl.sum(S * qt[:, None], axis=0)                             # [BV] reduce over k
        tl.store(y_ptr + v_off + rv, y, mask=mv)
        state = S.to(tl.int8)
        fire += tl.sum(tl.where(mk[:, None] & mv[None, :], S, 0))
    return fire


@triton.jit
def _recurrence_kernel(
    kt_ptr, vt_ptr, qt_ptr, wq_ptr, s_ptr, y_ptr, fire_ptr,
    B, T, K, V, theta, outer_gain,
    BK: tl.constexpr, BV: tl.constexpr,
):
    pid = tl.program_id(0)                       # one program per (p, b)
    p = pid // B
    pb = pid
    rv = tl.arange(0, BV)
    mv = rv < V
    # WqT[i,o] = Wq[p,o,i] : load transposed so tl.dot contracts the value index i.
    # Wq is (P,V,V) contiguous as [p,o,i]; tile [i (row, stride 1), o (col, stride V)].
    wq_base = wq_ptr + p * V * V
    wqT = tl.load(wq_base + rv[:, None] * 1 + rv[None, :] * V,
                  mask=mv[:, None] & mv[None, :], other=0).to(tl.int8)   # [BV(i), BV(o)]
    fire = _recur_body(kt_ptr, qt_ptr, vt_ptr, s_ptr, y_ptr, wqT, pb,
                       T, K, V, theta, outer_gain, BK, BV)
    tl.store(fire_ptr + pb, fire)


def triton_int_recurrence(kt, vt, qt, Wq, s, theta, return_fire=False, outer_gain=1):
    """Fast path mirroring `int_recurrence_reference` (same args, same result).

    kt (P,B,T,K) {0,1}; vt (P,B,T,V) {0,1}; qt (P,B,T,K) {0,1}; Wq (P,V,V) {-1,0,1};
    s (P,B,T) ints {0..3}; theta int; outer_gain int (k⊗v write weight, DESIGN §7).
    Returns Y (P,B,T,V) int32 in [0,K] (+ scalar firing rate if return_fire). State
    resets to 0 per call (per layer/seq).
    """
    P, Bn, T, K = kt.shape
    V = vt.shape[-1]
    dev = kt.device
    kt_i = kt.to(torch.int8).contiguous()
    vt_i = vt.to(torch.int8).contiguous()
    qt_i = qt.to(torch.int8).contiguous()
    wq_i = Wq.to(torch.int8).contiguous()
    s_i = s.to(torch.int32).contiguous()
    Y = torch.empty(P, Bn, T, V, dtype=torch.int32, device=dev)
    fire = torch.empty(P * Bn, dtype=torch.int32, device=dev)
    # int8 tl.dot requires the contraction dim (BV) >= 32; pad both tile dims to >=32.
    # The padding is masked to zero everywhere it's read, so the result is exact.
    BK = max(32, triton.next_power_of_2(K))
    BV = max(32, triton.next_power_of_2(V))
    grid = (P * Bn,)
    _recurrence_kernel[grid](
        kt_i, vt_i, qt_i, wq_i, s_i, Y, fire,
        Bn, T, K, V, int(theta), int(outer_gain),
        BK=BK, BV=BV,
    )
    if return_fire:
        return Y, fire.sum().item() / (P * Bn * T * K * V)
    return Y
