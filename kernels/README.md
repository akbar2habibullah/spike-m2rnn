# Stage 2 — Triton fused integer recurrence kernel

Implementation of the kernel specified in [DESIGN.md](DESIGN.md): the serial recurrence of
`SpikingM2RNN` (spike + MAC-free + fully-ternary) fused into one Triton program per
`(population member p, batch element b)`, carrying the matrix state `S` and the integer
membrane `mem` in registers/SRAM across time. The parallel projections stay in PyTorch
(DESIGN §1) — only the hot serial loop is fused.

## What's here
| File | Role |
|---|---|
| `../src/spiking_m2rnn/int_membrane.py` | **Integer-membrane reference** (Stage 2.0): the bit-exact contract — `int_recurrence_reference` + `ternary_W_materialize`. |
| `triton_recurrence.py` | **Stage 2.1**: fused integer/bitwise recurrence kernel (`int8` IMMA `tl.dot` + arithmetic shift + compare/reset), noise supplied from HBM. |
| `triton_philox.py` | **Stage 2.2**: in-SRAM Philox noise regeneration — `A_p,B_p` regenerated on-chip, `Wq=ternary(W+σABᵀ)` built in SRAM, used, discarded (P decoupled from HBM). |
| `test_kernel.py` | Stage 2.1 validation (§9.2): kernel ≡ reference, **bit-exact**, all shapes + end-to-end model logits. |
| `test_philox.py` | Stage 2.2 validation (§9.3 + §6 invariants): Gaussian / antithetic / decorrelated noise; regen recurrence ≡ same-noise reference; update≡forward noise. |
| `bench.py` | Throughput (kernel vs torch) and population-scaling (in-SRAM Philox keeps weight memory O(V²) as P grows). |

## The op mapping (DESIGN §3)
```
trans[k,o] = Σ_i state[k,i]·Wq[o,i]      int8 tl.dot (state binary, Wq ternary)  -> int32 [-V,V]
outer[k,o] = outer_gain · k_t[k]·v_t[o]  int8 outer product                       -> {0, gain}
mem        = (mem >> s_t) + trans + outer arithmetic right shift (s_t∈{0..3})
S          = mem > θ                     integer compare                          -> {0,1}
mem        = mem - θ·S                    subtractive reset
y_t[o]     = Σ_k S[k,o]·q_t[k]           int reduce over k                        -> [0,K]
```

### Two findings worth knowing
1. **TF32 / int8 are exact here.** `trans` and `y` are binary×ternary / binary×binary
   reductions over ≤64 terms — the operands are 0/±1, exactly representable, partial sums
   ≪ 2²⁴. So the reference computes them in fp32 (CUDA has no int matmul) and the kernel in
   `int8 tl.dot`, both bit-exact integers. Arithmetic `>>` on signed int32 = floor toward −∞,
   identical on both sides.
2. **The scale-fold is NOT a free identity (DESIGN §7).** Folding the per-member ternary
   `scale` (≈0.1) into θ leaves `outer ∈ {0,1}` ~10× too weak vs `trans` in the integer
   domain, so the network can't write key→value bindings. The fix is an integer
   `outer_gain ≈ round(1/scale) ≈ 10` (and θ to match); see `int_membrane.py`. With this the
   integer model learns at the same rate as the fp `mac_free+ternary_all` model (verified
   head-to-head).

## Run
```bash
python kernels/test_kernel.py     # Stage 2.1: bit-exact recurrence + end-to-end logits
python kernels/test_philox.py     # Stage 2.2: in-SRAM Philox invariants + bit-exact
python kernels/bench.py           # speedup + population scaling

# Train S3 with the kernel (Stage 2.0 re-validation + Stage 2.3 end-to-end):
python tasks/state_tracking/train_sn.py --n 3 --mode spike --int-membrane --use-kernel \
  --steps 3000 --train-lens 8 16 --eval-lens 16 32 64 128 \
  --pop 512 --batch 256 --dim 64 --depth 8 --k 32 --v 32 --mlp 64
```
`SpikingM2RNN(..., int_membrane=True, use_kernel=True)` runs the kernel; `use_kernel=False`
runs the torch reference (bit-identical, the equivalence target).

## Validation status
- **Stage 2.1** ✅ recurrence kernel **bit-exact** vs the integer reference across shapes
  (square 32/64, K≠V, non-power-of-2 with masking, θ∈{0,1,2,4,10}, outer_gain∈{1,10,11},
  T≤64), and the full model produces **bit-identical logits** torch-vs-kernel.
- **Stage 2.2** ✅ in-SRAM Philox: Gaussian (`tl.randn`), antithetic (±A pairing),
  decorrelated/deterministic; regen recurrence **bit-exact** vs same-noise reference
  (in-kernel ternary quantize matches torch with 0 element diffs); update regenerates
  bit-identical noise (invariant #1).
- **Throughput** ✅ ~84× faster recurrence than the torch reference at the proven scale;
  full training step 10 s → 0.83 s.
