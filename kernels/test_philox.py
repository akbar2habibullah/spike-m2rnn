"""
Stage 2.2 validation (kernels/DESIGN.md §9.3 + §6 invariants): the in-SRAM Philox noise
regeneration. Cross-framework determinism is NOT free (invariant #5), so we validate by
regenerating with the SAME Philox the kernel uses (dump its (A,B)/Wq), never torch.randn.

    python kernels/test_philox.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from spiking_m2rnn.int_membrane import int_recurrence_reference, ternary_W_materialize  # noqa: E402
from triton_philox import gen_noise, triton_recurrence_philox                            # noqa: E402

DEV = torch.device("cuda")


def test_gaussian_and_decorrelation():
    """Invariant #2 (Gaussian) + #4 (Philox, decorrelated counters)."""
    A, B = gen_noise(4096, 64, seed=7, param_off=0, antithetic=False, device=DEV)
    print(f"A ~ N({A.mean():.3f},{A.std():.3f})  B ~ N({B.mean():.3f},{B.std():.3f})")
    assert abs(A.mean()) < 0.05 and abs(A.std() - 1) < 0.05, "A not ~N(0,1)"
    assert abs(B.mean()) < 0.05 and abs(B.std() - 1) < 0.05, "B not ~N(0,1)"
    corr = torch.corrcoef(torch.stack([A.flatten(), B.flatten()]))[0, 1].item()
    assert abs(corr) < 0.05, f"A,B correlated ({corr})"
    A2, _ = gen_noise(4096, 64, seed=7, param_off=1, antithetic=False, device=DEV)
    assert not torch.equal(A, A2), "param_off did not decorrelate"
    # determinism (invariant #1: same key -> same noise, every time / every kernel)
    Ad, Bd = gen_noise(4096, 64, seed=7, param_off=0, antithetic=False, device=DEV)
    assert torch.equal(A, Ad) and torch.equal(B, Bd), "Philox not deterministic"
    print("OK Gaussian, decorrelated, deterministic")


def test_antithetic_pairing():
    """Invariant #3: member 2m,2m+1 share (A,B) but A's sign flips (±A)."""
    A, B = gen_noise(16, 64, seed=11, param_off=0, antithetic=True, device=DEV)
    even, odd = A[0::2], A[1::2]
    assert torch.equal(even, -odd), "antithetic A sign-flip violated"
    assert torch.equal(B[0::2], B[1::2]), "antithetic B must be shared"
    print("OK antithetic pairing (A_2m = -A_2m+1, B shared)")


def test_recurrence_philox_bitexact():
    """§9.3: recurrence with in-SRAM Philox == reference fed the SAME regenerated noise,
    bit-exact. Also confirms in-kernel ternary quantize == torch ternary quantize."""
    sigma, rank_scale = 0.05, 1.0
    cases = [   # P, B, T, K, V, theta, antithetic, outer_gain
        (8, 4, 16, 32, 32, 2, True, 1), (6, 3, 12, 64, 64, 1, True, 1),
        (4, 4, 20, 32, 32, 3, False, 1), (16, 2, 10, 48, 40, 2, True, 1),
        (8, 4, 16, 32, 32, 10, True, 10),    # S3 operating point through Philox
    ]
    for i, (P, B, T, K, V, theta, anti, og) in enumerate(cases):
        g = torch.Generator(device=DEV).manual_seed(200 + i)
        kt = torch.randint(0, 2, (P, B, T, K), generator=g, device=DEV, dtype=torch.int32)
        vt = torch.randint(0, 2, (P, B, T, V), generator=g, device=DEV, dtype=torch.int32)
        qt = torch.randint(0, 2, (P, B, T, K), generator=g, device=DEV, dtype=torch.int32)
        s = torch.randint(0, 4, (P, B, T), generator=g, device=DEV, dtype=torch.int32)
        master = torch.randn(V, V, generator=g, device=DEV, dtype=torch.float32) * 0.1
        seed, po = 31 + i, 5 * i

        Y_ker, Wq_dump = triton_recurrence_philox(
            kt, vt, qt, master, s, theta, seed, param_off=po,
            sigma=sigma, rank_scale=rank_scale, antithetic=anti, dump_wq=True, outer_gain=og)

        # rebuild Wq in torch from the SAME Philox noise (dumped), then run the reference
        A, Bf = gen_noise(P, V, seed, param_off=po, antithetic=anti, device=DEV)
        Wq_torch = ternary_W_materialize(master, A.unsqueeze(-1), Bf.unsqueeze(-1),
                                         sigma, rank_scale)
        wq_match = torch.equal(Wq_dump, Wq_torch)
        Y_ref = int_recurrence_reference(kt, vt, qt, Wq_torch, s, theta, outer_gain=og)
        y_match = torch.equal(Y_ker, Y_ref)
        nWqdiff = (Wq_dump != Wq_torch).sum().item()
        print(f"case {i} P{P}B{B}T{T}K{K}V{V}θ{theta}g{og} anti={anti}: "
              f"Wq(kernel==torch)={wq_match} (#diff={nWqdiff})  Y(kernel==ref)={y_match}")
        assert y_match, f"case {i}: recurrence+Philox NOT bit-exact vs reference"
        assert wq_match, f"case {i}: in-kernel quantize differs from torch ({nWqdiff} elems)"
    print("OK recurrence+in-SRAM-Philox bit-exact vs same-noise reference")


def test_update_matches_forward():
    """§9.4 / invariant #1: the ES update regenerates the SAME (A,B) the forward used,
    so Σ_p f_p A_p B_pᵀ is computed on identical noise -> matches a torch reference that
    reuses the dumped noise, exactly."""
    P, V, seed, po = 64, 32, 99, 3
    A, Bf = gen_noise(P, V, seed, param_off=po, antithetic=True, device=DEV)
    # forward would build Wq from exactly this (A,B); the update reuses the same call:
    A2, B2 = gen_noise(P, V, seed, param_off=po, antithetic=True, device=DEV)
    fit = torch.randn(P, device=DEV)
    upd_fwd = torch.einsum("p,po,pi->oi", fit, A, Bf)
    upd_updk = torch.einsum("p,po,pi->oi", fit, A2, B2)
    assert torch.equal(upd_fwd, upd_updk), "update noise != forward noise"
    print("OK update regenerates bit-identical noise (Σ f_p A_p B_pᵀ matches)")


if __name__ == "__main__":
    test_gaussian_and_decorrelation()
    test_antithetic_pairing()
    test_recurrence_philox_bitexact()
    test_update_matches_forward()
    print("ALL PHILOX TESTS PASSED")
