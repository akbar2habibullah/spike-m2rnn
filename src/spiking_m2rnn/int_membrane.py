"""
Stage 2.0 -- INTEGER-MEMBRANE recurrence reference (kernels/DESIGN.md §2).

The fp `mac_free` model leaks via `mem * 2^{-s}` (fractional membranes); the Triton
kernel will do an arithmetic right shift `mem >> s` (floor). These differ, so the
kernel CANNOT be bit-exact to the fp model. DESIGN §2 therefore mandates a true
*integer* recurrence as the kernel's correctness contract:

  trans = state(binary) · Wᵀ          ternary W∈{-1,0,+1}, state∈{0,1}  -> [-V, V]
  outer = k_t ⊗ v_t                    binary ⊗ binary                  -> {0,1}
  mem   = (mem >> s_t) + trans + outer arithmetic shift, s_t∈{0,1,2,3}
  S     = mem > θ                      integer threshold (swept)        -> {0,1}
  mem   = mem - θ·S                    subtractive reset
  y_t   = Sᵀ · q_t                     binary masked sum over k         -> [0, K]

This module is the materialized reference path (guardrail #2/#6): the Triton kernel
in `kernels/` is validated **bit-exact (exact integer equality)** against it.

Integer-matmul is unsupported on CUDA (`baddbmm_cuda not implemented for Int`), but
binary×ternary and binary×binary contractions over V,K≤64 are *exactly* representable
in fp32 (even TF32: inputs are 0/±1, partial sums ≤64 ≪ 2²⁴). So the two reductions
(`trans`, `y`) are done in fp32 and are bit-exact integers; all stateful arithmetic
(shift/add/compare/subtract) is genuine int32.
"""

import torch


def ternary_W_materialize(weight, A, B, sigma, rank_scale, eps=1e-5):
    """Build the per-member ternary transition matrix Wq ∈ {-1,0,+1} (DESIGN §4/§6).

    Mirrors `eggroll.ternary_quantize` (BitNet b1.58 absmean) on the EGGROLL-perturbed
    master `W_eff = weight + (σ/√r)·A Bᵀ`, but returns the *unscaled* ternary integers
    (the per-member absmean `scale` is a single scalar folded into θ downstream, DESIGN
    §7 -- `mem>θ ⇔ mem/scale > θ/scale`).

    weight (O,I) float master; A (P,O,r); B (P,I,r). Returns Wq (P,O,I) float in {-1,0,1}.
    """
    W_eff = weight.unsqueeze(0) + (sigma * rank_scale) * torch.einsum("por,pir->poi", A, B)
    scale = W_eff.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    Wq = torch.round(W_eff / scale).clamp_(-1.0, 1.0)
    return Wq                                            # (P,O,I) in {-1,0,+1}


@torch.no_grad()
def int_recurrence_reference(kt, vt, qt, Wq, s, theta, state0=None, return_fire=False,
                             outer_gain=1, fp_bits=0, round_shift=False):
    """Bit-exact integer recurrence over time (the kernel contract).

    Shapes (P=pop, B=batch, T=time, K=key dim, V=value dim):
      kt : (P,B,T,K) binary {0,1}      key spikes
      vt : (P,B,T,V) binary {0,1}      value spikes
      qt : (P,B,T,K) binary {0,1}      query spikes
      Wq : (P,V,V)   ternary {-1,0,1}  trans[k,o] = Σ_i state[k,i]·Wq[o,i]
      s  : (P,B,T)   int {0,1,2,3}     per-step shift-decay amount
      theta : int                      integer firing threshold
      outer_gain : int                 integer weight on the k⊗v write (DESIGN §7 fix)
      fp_bits : int                    fixed-point fractional bits on the membrane (see below)
      round_shift : bool               round-to-nearest leak instead of floor (see below)
      state0 : (P,B,K,V) int32 binary or None (zeros)
    Returns Y (P,B,T,V) int32 in [0,K]; if return_fire also the scalar firing rate.

    `outer_gain`: the naive scale-fold (outer∈{0,1}, θ small) is NOT equivalent to the fp
    model (DESIGN §7 "verify, don't assume"). The fp transition is `scale·trans_int`
    (scale≈0.1), so in integer units the k⊗v write and θ both carry ≈1/scale≈10. Folding
    only θ leaves `outer` ~10× too weak to write key→value bindings -> the net can't track
    state. `outer_gain≈round(1/scale)` (and θ to match) restores the fp balance.

    fp→int RESOLUTION (the two knobs that recover the last few % vs the fp model):
      * `fp_bits=F`  -- run the membrane in fixed-point units of 2^{-F}: scale the per-step
        contributions and θ by 2^F so the leak `>>s` keeps F fractional bits of the decayed
        value instead of flooring to integers. F=0 is the pure-integer floor model; F→∞
        recovers the fp `mac_free` dynamics. Still pure integer/shift (kernel-friendly),
        just with headroom. Typically F=3-4 closes the gap.
      * `round_shift` -- round-to-nearest arithmetic shift `(mem + 2^{s-1}) >> s` instead of
        truncation; cheaply removes the downward bias of the floor.
    Both default OFF so the existing kernel (floor, no fp_bits) stays bit-exact at defaults;
    once a config reaches 100% on the reference, the Triton kernel must be updated to match.

    The trans/y reductions go through fp32 (exact for these integer ranges) so the
    function runs on CUDA; everything else is int32.
    """
    P, B, T, K = kt.shape
    V = vt.shape[-1]
    dev = kt.device
    Wqf = Wq.to(torch.float32)                            # {-1,0,1}, exact in fp32
    if state0 is None:
        state = torch.zeros(P, B, K, V, dtype=torch.int32, device=dev)
    else:
        state = state0.to(torch.int32)
    mem = torch.zeros(P, B, K, V, dtype=torch.int32, device=dev)
    scale = 1 << int(fp_bits)                            # fixed-point unit (2^F)
    theta_i = int(theta) * scale                         # θ in fixed-point units
    og = int(outer_gain)
    ys = []
    fire_acc, fire_n = 0.0, 0
    for t in range(T):
        kt_t = kt[:, :, t, :].to(torch.float32)          # (P,B,K) {0,1}
        vt_t = vt[:, :, t, :].to(torch.float32)          # (P,B,V)
        qt_t = qt[:, :, t, :].to(torch.float32)          # (P,B,K)
        # outer = outer_gain · (k_t ⊗ v_t)  ∈ {0, outer_gain}
        outer = (torch.einsum("pbk,pbv->pbkv", kt_t, vt_t).round().to(torch.int32)) * og
        # trans[k,o] = Σ_i state[k,i]·Wq[o,i]  ∈ [-V, V]  (exact integer via fp32)
        trans = torch.einsum("pbki,poi->pbko", state.to(torch.float32), Wqf)
        trans = trans.round().to(torch.int32)
        s_t = s[:, :, t].to(torch.int32)[:, :, None, None]   # (P,B,1,1)
        if round_shift:                                  # round-to-nearest leak: + 2^{s-1}
            radd = torch.where(s_t > 0,
                               torch.bitwise_left_shift(torch.ones_like(s_t), (s_t - 1).clamp(min=0)),
                               torch.zeros_like(s_t))
            leaked = torch.bitwise_right_shift(mem + radd, s_t)
        else:
            leaked = torch.bitwise_right_shift(mem, s_t)
        mem = leaked + (trans + outer) * scale           # contributions in fixed-point units
        S = (mem > theta_i).to(torch.int32)              # (P,B,K,V) {0,1}
        mem = mem - theta_i * S
        # y_t[v] = Σ_k S[k,v]·q_t[k]  ∈ [0, K]   (binary, unaffected by fp_bits)
        y = torch.einsum("pbkv,pbk->pbv", S.to(torch.float32), qt_t).round().to(torch.int32)
        ys.append(y)
        state = S
        if return_fire:
            fire_acc += S.float().mean().item(); fire_n += 1
    Y = torch.stack(ys, dim=2)                            # (P,B,T,V) int32
    if return_fire:
        return Y, fire_acc / max(fire_n, 1)
    return Y
