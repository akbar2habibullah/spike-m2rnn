"""
EGGROLL evolution-strategies machinery -- the canonical ES math, mirrored verbatim
from the working MNIST ViT (`reference/eggroll_vit_mnist.py`). Do not let this drift
from the ViT: it is the path verified equivalent to brute-force weight
materialization to ~1e-15 in float64.

What lives here:
  * `eggroll_linear` / `eggroll_ln` -- memory-efficient forward: ONE shared base GEMM
    plus a cheap low-rank correction `(sigma/sqrt r) * (x @ B) @ A^T`; a perturbed
    weight is NEVER materialized per population member.
  * `sample_noise` / `zero_noise` -- per-member perturbations keyed off a
    `ParameterDict`: a `_w` matrix gets low-rank `(A, B)` factors, everything else
    gets dense N(0,1) noise.
  * `per_member_loss` / `fitness_from_loss` -- rank-based fitness shaping.
  * `es_update` -- the in-place parameter step `coeff * sum_p f_p A_p B_p^T`.

Convention (DESIGN / CLAUDE.md): key suffix `_w` => matrix => low-rank; any other
key => dense.
"""

import torch
import torch.nn.functional as F

from . import config


# ===================== memory-efficient EGGROLL ops =====================
def eggroll_linear(x, weight, A, B, bias=None, bias_noise=None,
                   sigma=config.SIGMA, rank_scale=config.RANK_SCALE):
    """Shared base GEMM + cheap low-rank correction.

    x: (B,S,I) -> (P,B,S,O)   [first layer: a new pop dim is broadcast in]
       or (P,B,S,I) -> (P,B,S,O)   [later layers: pop dim already present]
    A:(P,O,r)  B:(P,I,r)  bias:(O,) shared  bias_noise:(P,O) raw N(0,1)
    """
    base = F.linear(x, weight)                       # shared base; one big GEMM either way
    if x.dim() == 3:                                 # (B,S,I): broadcast a new pop dim
        lr = torch.einsum("bsi,pir->bspr", x, B)
        lr = torch.einsum("bspr,por->pbso", lr, A)
        out = base.unsqueeze(0) + (sigma * rank_scale) * lr
    else:                                            # (P,B,S,I): pop dim already there
        lr = torch.einsum("pbsi,pir->pbsr", x, B)
        lr = torch.einsum("pbsr,por->pbso", lr, A)
        out = base + (sigma * rank_scale) * lr
    if bias is not None:
        out = out + bias                             # broadcast over (P,B,S)
    if bias_noise is not None:
        out = out + sigma * bias_noise[:, None, None, :]
    return out                                       # (P,B,S,O)


def ternary_quantize(W, eps=1e-5):
    """BitNet b1.58 absmean ternary: per-(member,tensor) scale = mean(|W|), then
    round(W/scale) clamped to {-1,0,+1}, returned as scale * {-1,0,+1}. W: (..., O, I)."""
    scale = W.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    Wq = torch.round(W / scale).clamp_(-1.0, 1.0)
    return Wq * scale


def ternarize_factor(t):
    """Deterministic ternary {-1,0,+1} of a low-rank noise factor, used by the
    no-materialize ternary path (`eggroll_linear_ternary_lowrank`). round(clamp(·)) ->
    ~38% zeros, rest ±1 for N(0,1) input. Applied IDENTICALLY in the forward and in
    `es_update`, so the perturbation that is applied is exactly the one accumulated
    (the ES invariant). The resulting ES perturbation is ternary/Rademacher rather than
    Gaussian -- a different (SPSA-like) search distribution; re-validate (DESIGN §6.6,
    invariant #2)."""
    return torch.round(t.clamp(-1.0, 1.0))


def eggroll_linear_ternary_lowrank(x, weight, A, B, bias=None, bias_noise=None,
                                   sigma=config.SIGMA, rank_scale=config.RANK_SCALE):
    """Multiply-free EGGROLL linear that STAYS LINEAR -> no per-member materialization.

    The §6.6 tension is only there because `quantize(W + sigma A Bᵀ)` is nonlinear. If we
    instead ternarize the *base* once (shared, O(O·I)) and the *factors* (ternary low-rank),
    the forward is `ternary_base @ x + sigma·rank_scale·(x @ Bt) @ Atᵀ` -- LINEAR in the
    perturbation, so EGGROLL's never-materialize trick is recovered (one shared base GEMM +
    a cheap low-rank correction, O((O+I)·r) per member instead of O(O·I)). Both terms are
    ternary×activation => add-only (BitNet), so it is also multiply-free (up to the per-matrix
    absmean scalar and the sigma scale).

    Trade-off vs `eggroll_linear_ternary`: the ES perturbation is now ternary, not Gaussian,
    and exploration happens via the additive low-rank term rather than by flipping the base's
    quantization (re-validate -- DESIGN §6.2/§6.6). `weight` is the float latent master,
    quantized to the ternary base each forward (QAT-style); ES updates the master via
    `Σ f_p ternarize(A_p) ternarize(B_p)ᵀ` (see `es_update(..., ternarize_keys=...)`).

    x: (P,B,S,I) or (B,S,I); weight (O,I); A (P,O,r); B (P,I,r)."""
    Wt = ternary_quantize(weight)                    # shared ternary base, quantized ONCE
    At = ternarize_factor(A)                          # (P,O,r) ternary
    Bt = ternarize_factor(B)                          # (P,I,r) ternary
    base = F.linear(x, Wt)                            # ONE shared GEMM (ternary x activation)
    if x.dim() == 3:                                  # (B,S,I): broadcast a new pop dim
        lr = torch.einsum("bsi,pir->bspr", x, Bt)
        lr = torch.einsum("bspr,por->pbso", lr, At)
        out = base.unsqueeze(0) + (sigma * rank_scale) * lr
    else:                                             # (P,B,S,I): pop dim already there
        lr = torch.einsum("pbsi,pir->pbsr", x, Bt)
        lr = torch.einsum("pbsr,por->pbso", lr, At)
        out = base + (sigma * rank_scale) * lr
    if bias is not None:
        out = out + bias
    if bias_noise is not None:
        out = out + sigma * bias_noise[:, None, None, :]
    return out                                        # (P,B,S,O)


def eggroll_linear_ternary(x, weight, A, B, bias=None, bias_noise=None,
                           sigma=config.SIGMA, rank_scale=config.RANK_SCALE, quantize=True):
    """Ternary EGGROLL linear (Stage 1b). Ternary breaks the no-materialize trick
    (quantize(W + sigma A B^T) is not linear in the perturbation), so each member's
    perturbed weight IS materialized, quantized, and used -- no shared base GEMM
    (DESIGN 6.6). `weight` is the FLOAT latent master that ES still updates via the
    usual sum_p f_p A_p B_p^T rule (quantization is forward-only).

    x: (P,B,S,I) or (B,S,I); weight (O,I); A (P,O,r); B (P,I,r).
    With quantize=False this is numerically identical to `eggroll_linear` -- kept as the
    materialized reference path for the equivalence test (guardrail #2)."""
    W_eff = weight.unsqueeze(0) + (sigma * rank_scale) * torch.einsum("por,pir->poi", A, B)  # (P,O,I)
    if quantize:
        W_eff = ternary_quantize(W_eff)
    if x.dim() == 3:                                 # (B,S,I): broadcast the pop dim
        out = torch.einsum("bsi,poi->pbso", x, W_eff)
    else:                                            # (P,B,S,I)
        out = torch.einsum("pbsi,poi->pbso", x, W_eff)
    if bias is not None:
        out = out + bias
    if bias_noise is not None:
        out = out + sigma * bias_noise[:, None, None, :]
    return out


def eggroll_ln(x, g_base, g_noise, b_base, b_noise, sigma=config.SIGMA, eps=1e-5):
    # x:(P,B,S,D). gains/biases are perturbed too: g,b become (P,D).
    mu  = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    xn  = (x - mu) * torch.rsqrt(var + eps)
    g   = g_base + sigma * g_noise                   # (P,D)
    b   = b_base + sigma * b_noise
    return xn * g[:, None, None, :] + b[:, None, None, :]


# ===================== per-member noise =====================
@torch.no_grad()
def sample_noise(params, pop, rank, device, dtype):
    """Per-member perturbation for each base parameter in `params` (a ParameterDict).

    `_w` matrices -> low-rank (A, B) factor tuple; everything else -> dense N(0,1).
    """
    noise = {}
    for name, par in params.items():
        if name.endswith("_w"):                      # linear weight -> low-rank factors
            o, i = par.shape
            noise[name] = (torch.randn(pop, o, rank, device=device, dtype=dtype),
                           torch.randn(pop, i, rank, device=device, dtype=dtype))
        else:                                        # bias / LN gain / etc -> dense
            noise[name] = torch.randn(pop, *par.shape, device=device, dtype=dtype)
    return noise


@torch.no_grad()
def zero_noise(params, rank, device, dtype):
    """Noise that makes the forward use the base/mean parameters (population of 1)."""
    noise = {}
    for name, par in params.items():
        if name.endswith("_w"):
            o, i = par.shape
            noise[name] = (torch.zeros(1, o, rank, device=device, dtype=dtype),
                           torch.zeros(1, i, rank, device=device, dtype=dtype))
        else:
            noise[name] = torch.zeros(1, *par.shape, device=device, dtype=dtype)
    return noise


# ===================== ES loss / fitness =====================
def per_member_loss(logits, targets):
    P, B, T, C = logits.shape
    flat = logits.reshape(P * B * T, C).float()
    tgt  = targets.reshape(B * T).repeat(P)          # tile (B*T) block P times -> p-major flatten
    return F.cross_entropy(flat, tgt, reduction="none").reshape(P, B * T).mean(1)   # (P,)


def fitness_from_loss(loss):
    ranks = torch.argsort(torch.argsort(loss, descending=True))
    return ranks.float() / (loss.shape[0] - 1) - 0.5


# ===================== parameter update =====================
def newton_schulz(G, steps=5, eps=1e-7):
    """Muon's Newton-Schulz quintic iteration: returns the approximate orthogonal /
    polar factor of a 2D matrix G (i.e. U V^T for G = U S V^T), equalizing the singular
    values to ~1 while preserving the update's orientation. Computed in float32."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)                          # bound the spectral norm <= 1
    transposed = X.shape[0] > X.shape[1]
    if transposed:                                    # iterate on the smaller dimension
        X = X.t()
    for _ in range(steps):
        Z = X @ X.t()
        X = a * X + (b * Z + c * (Z @ Z)) @ X
    return X.t() if transposed else X


@torch.no_grad()
def es_update(params, noise, fit, coeff, rank_scale=config.RANK_SCALE,
              muon=False, muon_lr=0.02, ns_steps=5, eps=1e-7, ternarize_keys=None):
    """In-place ES step.

    Default (muon=False): par += coeff * upd, where for `_w` matrices
    upd = (1/sqrt r) sum_p f_p A_p B_p^T (no perturbed weight materialized), and for
    dense params upd = sum_p f_p * noise_p (tensordot over the pop axis).

    `ternarize_keys`: set of `_w` names trained with `eggroll_linear_ternary_lowrank`. For
    these the perturbation that was *applied* used `ternarize_factor(A/B)`, so the update
    must accumulate the SAME ternarized factors (else forward != update -> biased estimator).

    muon=True: orthogonalize the matrix update (Newton-Schulz) and RMS-normalize the
    vector update, then step by `muon_lr`. This DECOUPLES the step magnitude from the
    gradient-estimate magnitude -- so POP only sharpens the direction, it no longer
    shrinks the step (the fix for "larger POP -> slower convergence" on the spiking /
    ternary dead-zone landscape, DESIGN 6.2). `coeff` is unused in this mode."""
    for name, par in params.items():
        if name.endswith("_w"):
            A, B = noise[name]
            if ternarize_keys is not None and name in ternarize_keys:
                A, B = ternarize_factor(A), ternarize_factor(B)
            upd = rank_scale * torch.einsum("p,por,pir->oi", fit, A, B)
        else:
            upd = torch.tensordot(fit, noise[name], dims=([0], [0]))

        if muon:
            if upd.dim() == 2:                        # matrix: orthogonalize (unit sing. values)
                X = newton_schulz(upd, ns_steps, eps)
                step = muon_lr * (max(upd.shape) ** 0.5) * X      # per-element RMS ~ muon_lr
            else:                                     # vector: RMS-normalize to ~1 per element
                g = upd.float()
                step = muon_lr * g / (g.norm() + eps) * (g.numel() ** 0.5)
            par.data.add_(step.to(par.dtype))
        else:
            par.data.add_(coeff * upd.to(par.dtype))
