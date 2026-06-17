"""
Stage 2.1 validation (kernels/DESIGN.md §9.2): the Triton recurrence kernel must be
BIT-EXACT (exact integer equality, not a tolerance) vs the integer-membrane reference
`int_recurrence_reference`, on random inputs across all shapes -- and, end-to-end, the
full `SpikingM2RNN` int_membrane forward must produce bit-identical logits whether the
recurrence runs in torch or the kernel.

    python kernels/test_kernel.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from spiking_m2rnn.int_membrane import int_recurrence_reference   # noqa: E402
from spiking_m2rnn.model import SpikingM2RNN                       # noqa: E402
from triton_recurrence import triton_int_recurrence               # noqa: E402

DEV = torch.device("cuda")


def _rand_case(P, B, T, K, V, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    kt = torch.randint(0, 2, (P, B, T, K), generator=g, device=DEV, dtype=torch.int32)
    vt = torch.randint(0, 2, (P, B, T, V), generator=g, device=DEV, dtype=torch.int32)
    qt = torch.randint(0, 2, (P, B, T, K), generator=g, device=DEV, dtype=torch.int32)
    Wq = torch.randint(-1, 2, (P, V, V), generator=g, device=DEV, dtype=torch.int32)
    s = torch.randint(0, 4, (P, B, T), generator=g, device=DEV, dtype=torch.int32)
    return kt, vt, qt, Wq, s


def test_recurrence_bitexact():
    # cover: square 32/64, K≠V, non-power-of-2 (masking), θ∈{0,1,2,4,10}, outer_gain∈{1,10}
    cases = [   # P, B, T, K, V, theta, outer_gain
        (4, 3, 8, 32, 32, 2, 1),
        (8, 5, 16, 32, 32, 1, 1),
        (4, 4, 16, 64, 64, 2, 1),
        (6, 2, 12, 64, 32, 3, 1),      # K≠V
        (3, 3, 10, 48, 40, 2, 1),      # non-power-of-2 -> exercises masking
        (5, 4, 64, 32, 32, 4, 1),      # long T, larger θ
        (4, 3, 8, 32, 32, 0, 1),       # θ=0 (no-op reset / leaky integrator)
        (2, 2, 6, 16, 16, 1, 1),       # min tl.dot tile
        (8, 4, 16, 32, 32, 10, 10),    # the S3 operating point (θ=outer_gain=10)
        (4, 3, 12, 48, 40, 10, 11),    # outer_gain≠θ, non-pow2
    ]
    worst = 0
    for i, (P, B, T, K, V, th, og) in enumerate(cases):
        kt, vt, qt, Wq, s = _rand_case(P, B, T, K, V, seed=100 + i)
        ref = int_recurrence_reference(kt, vt, qt, Wq.float(), s, th, outer_gain=og)
        ker = triton_int_recurrence(kt, vt, qt, Wq, s, th, outer_gain=og)
        assert ref.shape == ker.shape == (P, B, T, V)
        diff = (ref - ker).abs().max().item()
        worst = max(worst, diff)
        ok = torch.equal(ref, ker)
        assert ker.min() >= 0 and ker.max() <= K, f"case {i}: y out of [0,{K}]"
        print(f"case {i} P{P}B{B}T{T}K{K}V{V}θ{th}g{og}: bit-exact={ok} maxdiff={diff} "
              f"y∈[{ker.min().item()},{ker.max().item()}] fired={(ker>0).float().mean():.2f}")
        assert ok, f"case {i}: NOT bit-exact (maxdiff={diff})"
    print(f"OK recurrence bit-exact across {len(cases)} cases (worst maxdiff={worst})")


def test_model_forward_bitexact():
    """End-to-end: full int_membrane model, recurrence in torch vs kernel -> identical logits."""
    torch.manual_seed(0)
    vocab = 6
    m = SpikingM2RNN(vocab, dim=64, depth=4, k=32, v=32, mlp=64, mode="spike",
                     int_membrane=True, theta=10).to(DEV).to(torch.float16)
    m.eval(); m.requires_grad_(False)
    idx = torch.randint(0, vocab, (4, 24), device=DEV)
    noise = m.sample_noise(8, 1, DEV, torch.float16)

    m.use_kernel = False
    lo_ref = m(idx, noise, 0.05)
    m.use_kernel = True
    lo_ker = m(idx, noise, 0.05)
    assert torch.equal(lo_ref, lo_ker), \
        f"model logits differ: maxabs={ (lo_ref-lo_ker).abs().max().item() }"
    print(f"OK end-to-end model logits bit-identical (torch vs kernel), shape={tuple(lo_ker.shape)}")


if __name__ == "__main__":
    test_recurrence_bitexact()
    test_model_forward_bitexact()
    print("ALL KERNEL EQUIVALENCE TESTS PASSED")
