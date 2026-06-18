# Stage 2 — Triton kernel design

> **Audience:** a fresh Claude Code instance running on a GPU. This is the implementation
> spec for the Stage-2 fused kernel. Read [docs/DESIGN.md](../docs/DESIGN.md) §6–§7 and
> [CLAUDE.md](../CLAUDE.md) first for project intent and guardrails; this file is the
> kernel-specific contract. The reference (correctness) implementation is the PyTorch
> model in [src/spiking_m2rnn/model.py](../src/spiking_m2rnn/model.py) (spike mode with
> `mac_free=True, ternary_all=True`) and the EGGROLL ops in
> [src/spiking_m2rnn/eggroll.py](../src/spiking_m2rnn/eggroll.py).

## 0. Status entering Stage 2 (what's already proven)
The full multiply-free model **solves S3 state-tracking with perfect length generalization**
(100% `pos@L128`, trained at length ≤16) in pure PyTorch. Stage 2 changes **throughput, not
behavior**: it makes the fully-ternary forward fast and lets population `P` scale without
HBM pressure. It must not change what the model computes (within the integer-semantics
caveat in §2).

## 1. Scope — what the kernel accelerates vs. what stays in PyTorch

The model has two regions (see `forward` in model.py):

- **Serial recurrence** (the `for t in range(T)` loop): sequential, runs T times, and is
  **already fully integer/bitwise** in spike+mac_free+ternary mode. **This is the kernel's
  hot path.** Fuse the whole per-step update into one Triton kernel.
- **Parallel projections** (`embed`, `q/k/v/o/fc1/fc2/head`, LayerNorm, GELU): computed
  once across the sequence, amortized over T. **Leave these in PyTorch / cuBLAS-style
  int8 for now.** They are ternary-weight × *float*-activation (not popcount-able) and are
  not latency-critical. Optimizing them is a later, optional pass (§10).

So: **Stage 2 = a fused, integer/bitwise, ternary-transition recurrence kernel** (plus the
EGGROLL in-SRAM noise machinery, §6). Everything else stays in the existing PyTorch path.

## 2. The bit-exactness target — READ THIS FIRST (likely the biggest gotcha)

Guardrail #1 says "Stage 2 must be bit-exact vs Stage 1." But the Stage-1 float model is
**not** integer: `mac_free` computes the leak as `mem * 2^{-s_t}` in fp, which produces
**fractional** membranes (e.g. `3 * 0.5 = 1.5`), whereas the kernel will do an arithmetic
**right shift** `mem >> s_t` (floor, `3 >> 1 = 1`). These differ. The threshold `mem > θ`
then diverges. **The kernel cannot be bit-exact to the current fp `mac_free` model.**

**Required first step (do this before any Triton):** add an **integer-membrane reference**
to the PyTorch model — a spike+mac_free variant that uses true integer ops (int16/int32
membrane, arithmetic right shift, integer `trans` from ternary×binary, integer threshold,
subtractive reset). Concretely:
- `trans` = integer matmul of ternary `W ∈ {−1,0,+1}` and binary `state ∈ {0,1}` → values in `[−V, V]`.
- `outer` = `k_t ⊗ v_t ∈ {0,1}`.
- `mem = (mem >> s_t) + trans + outer` (arithmetic shift; define behavior for negative `mem`).
- `S = mem > θ` with **integer** `θ` (re-pick; the fp `θ=1.0` becomes an integer threshold — sweep small values and re-validate S3).
- `mem = mem - θ * S`.
- `y_t[v] = Σ_k S[k,v] * q_t[k]` (integer in `[0, K]`).

Then:
1. **Re-validate** that the integer-membrane model still solves S3 length-gen (it may need a
   different `θ` / `s_t` mapping; the fp→int transition can shift dynamics). This is a
   Stage-1.5 checkpoint — do not proceed to Triton until S3 is green on the integer model.
2. The Triton kernel is **bit-exact vs this integer-membrane PyTorch reference** (the
   "materialized reference path," guardrail #2/#6), validated in int (exact, not a tolerance).

Keep the fp `mac_free` model as the algorithmic prototype; the integer model is the kernel's
contract.

**2a. Resolution knobs (the fp→int gap, and what closes it).** The pure-integer floor model
(`mem >> s`) loses the fractional membrane precision the fp FSA relied on, and converges but
**stalls short of 100%**. Two reference-side knobs recover it (both in
`int_recurrence_reference`, opt-in via `--fp-bits` / `--round-shift`, default off so the
current kernel stays bit-exact):
- **`fp_bits=F`** — run the membrane in fixed-point units of `2^{-F}`: contributions and `θ`
  scale by `2^F`, so the leak `>>s` keeps F fractional bits instead of flooring. `F=0` is the
  floor model, `F→∞` recovers fp. Typically `F=3–4` closes the gap. Still pure integer/shift.
- **`round_shift`** — round-to-nearest leak `(mem + 2^{s-1}) >> s` instead of truncation.

**Workflow:** sweep `--fp-bits`/`--round-shift` on the *reference* (`use_kernel=False`) until
S3 hits 100%; lock that `(F, round_shift, θ, outer_gain)` config; **then update the Triton
kernel to implement the same fixed-point scaling + rounding** and re-confirm bit-exactness.
The model raises if `use_kernel=True` with `F>0`/`round_shift` set (the kernel doesn't
implement them yet) — don't run mismatched semantics.

## 3. Op-by-op dispatch table (recurrence step)

| Op (per timestep, per (p,b)) | Math | Kernel realization | Notes |
|---|---|---|---|
| `outer = k_t ⊗ v_t` | binary ⊗ binary → (K×V) | AND of bit-packed `k_t`,`v_t` | spikes packed as bitmasks |
| `trans = state · Wᵀ` | ternary (V×V) × binary (K×V) → (K×V) int | **bitplane popcount** (§4) or int8-IMMA | the core op |
| leak `mem >> s_t` | arithmetic right shift | integer shift; `s_t∈{0,1,2,3}` scalar per (p,b) | s_t from gate (§3a) |
| `mem += trans + outer` | int add | integer add | int16/int32 accumulator (§7) |
| `S = mem > θ` | compare | integer compare → bitmask | this is the "activation" — just a compare |
| reset `mem -= θ·S` | int subtract | masked subtract | subtractive reset |
| `y_t = Sᵀ · q_t` | binary masked sum over k | popcount/AND-reduce per v | integer in [0,K] |

**3a. The decay gate `s_t`.** In the reference it's `s_t = round(sigmoid(raw)·3)` from a
projection `raw`. At inference this is a **step function** → precompute the 3 cutoffs on
`raw` and emit `s_t∈{0,1,2,3}` with **comparisons** (no sigmoid, no LUT). `raw` comes from
the parallel `_d` projection.

**3b. Pointwise nonlinearities elsewhere.** GELU (in the MLP) → 256-entry **LUT** on
quantized int8 activations if/when the MLP is quantized; otherwise fp16. The spike encodings
(`q/k/v = proj > 0`) are **comparisons**. None of these are in the serial loop.

**3c. LayerNorm.** A *reduction* (mean/var/rsqrt), not pointwise → not LUT-able. Keep fp16 in
the parallel path (it's amortized; does not undermine the multiply-free claim — DESIGN §6.5).
Optional later: RMSNorm + `rsqrt` via LUT+Newton.

## 4. Data layout & bitplane format

- **Ternary weight `W` (V×V), per member:** two bitplanes, `Wpos` (bit set where `W=+1`) and
  `Wneg` (bit set where `W=−1`). Pack each row's V entries into `ceil(V/32)` `uint32` (V=64 →
  2×uint32, or 1×uint64). Built **in SRAM** from the regenerated perturbation (§6), reused
  across all `t` and all `b` for that member.
- **Binary spikes (`S`, `k_t`, `v_t`, `q_t`):** bit-packed vectors, V or K bits → `uint32`
  words. `state = S` carried in registers/SRAM across `t`.
- **Membrane `mem` (K×V):** integer (int16 default; see §7), in SRAM.
- **Transition via popcount:** `trans[k,o] = popcount(state[k,:] & Wpos[o,:]) −
  popcount(state[k,:] & Wneg[o,:])`. Two popcounts per output over the bit-packed V vector.

**Packing choices (pick per the GPU and Triton support):**
- **SWAR popcount** (`&` + `tl`-level popcount): word-width parallelism (32/word), works at
  any V, no tensor cores. Simplest; good for small V=32/64.
- **int8 IMMA** (`tl.dot(int8)→int32`): store ternary as int8 {−1,0,1}, binary as int8;
  tensor cores. Easiest in Triton, batches the P·B·K axes into tiles (§5).
- **1-bit (b1) tensor cores** (sm_75+, XOR/AND+popcount): densest (~32–64× fp16) but Triton
  sub-byte support is experimental — treat as a stretch optimization, not the first target.

Recommended first target: **int8 IMMA** for `trans` (cleanest in Triton, uses tensor cores,
batches P·B·K), with **SWAR popcount** as the fallback/validation cross-check. Note `W` is
tiny (V≤64), so a single matvec underfills tiles — throughput comes from batching P·B·K (§5).

## 5. Parallelization & tiling

- **Parallel axes (no dependency):** population `P`, batch `B`, and the key index `K` (the K
  rows of the matrix state are independent in `trans` and the membrane update). Map `P×B` (and
  optionally `K`) to the CUDA grid / Triton program ids.
- **Serial axis:** time `T` (carry `state`, `mem` in SRAM across the loop).
- **Reuse:** load/build `Wpos,Wneg` for a member once, reuse across all `t,b`. The base `W`
  master is shared across members — load once per block of members.
- For int8-IMMA, accumulate the `trans` matmul over the P·B·K batch into tensor-core tiles;
  for SWAR, one program per (p,b) (or (p,b,k-tile)) looping t.

## 6. EGGROLL in-SRAM noise (the hyperscale trick) — DESIGN §7 invariants

Instead of storing per-member `(A_p, B_p)` low-rank factors in HBM, **regenerate them in
SRAM** from an integer seed keyed by `(step, member p, param_id)`, build `W_eff_p =
ternary(W_master + (σ/√r)·A_p B_pᵀ)` on-chip, use, discard. This decouples `P` from HBM
capacity/bandwidth → `P` becomes a pure time knob.

**Hard invariants (violating any silently biases the ES estimator):**
1. **Forward and update kernels regenerate bit-identical noise** — same Philox key `(step,p)`,
   distinct per-parameter counter offsets. Agree by construction.
2. **Gaussian factors** — EGGROLL samples `randn`; the update's `1/σ` scaling assumes `N(0,I)`.
   Add a uniform→normal transform (Box–Muller / inverse-CDF) in-kernel; raw Philox is uniform.
3. **Antithetic pairing baked into the seed map** — even/odd member ⇒ `±A` (variance reduction).
4. **Philox, not LCG**; distinct counter offsets so `A`, `B`, and different matrices decorrelate.
5. **Triton-Philox ≠ `torch.randn`.** Cross-framework determinism is NOT free. The validation
   path must regenerate noise *with the same Philox* in a torch reference (small P); production
   trusts kernel↔kernel self-consistency only.
6. **Keep the materialized reference** (§2): a torch path that builds `W_eff` from the
   *same* Philox noise and runs the *integer* recurrence, for bit-exact checks.

## 7. Numerical precision

- `trans ∈ [−V, V]` (V≤64 → [−64,64]); `outer ∈ {0,1}`; subtractive reset keeps `mem` bounded
  near `θ`. **int16 membrane** is safe; int8 may suffice if `θ` and reset keep range small —
  validate the actual range on S3/S5 before narrowing.
- Arithmetic right shift on **signed** `mem` must match the torch reference's shift semantics
  exactly (floor toward −∞ vs toward 0 — pick one, implement identically both sides).
- The per-member ternary `scale` (BitNet absmean, `ternary_quantize` in eggroll.py) is a float
  per (member, matrix); fold it into the downstream accumulation or keep `trans` in scaled-int.
  For the **transition** specifically, the scale is a single per-member scalar → can be applied
  once to `θ` instead of to every `trans` element (algebraically equivalent: `mem>θ` ⇔
  `mem/scale > θ/scale`). Confirm this simplification against the reference.

## 8. The update path
ES update on the **float master** weights is unchanged: `W_master += coeff · Σ_p f_p A_p B_pᵀ`
(or the Muon-orthogonalized variant — keep Muon/Newton-Schulz in torch on the small masters,
it's cheap; see `es_update`/`newton_schulz` in eggroll.py). The update kernel (or torch)
regenerates the **same** `(A_p,B_p)` via Philox (invariant #1) and accumulates the
fitness-weighted outer products. Quantization is forward-only; the master stays float.

## 9. Validation plan (in order; each gates the next)
1. **Integer-membrane torch reference** (§2) reproduces S3 length-gen. *Checkpoint: S3 100%.*
2. **Recurrence kernel, noise supplied from torch** (no Philox yet): bit-exact (int, exact
   equality) vs the integer torch reference on random inputs, all shapes.
3. **+ in-SRAM Philox**: forward kernel vs a torch path using the *same* Philox noise (not
   `torch.randn`) — bit-exact.
4. **Update kernel**: regenerated noise matches forward (invariant #1); `Σ f_p A_p B_pᵀ`
   matches the torch reference.
5. **End-to-end**: kernel-trained S3 reproduces the PyTorch length-gen result; then benchmark
   throughput vs PyTorch and scale `P`.

## 10. Stage-2 sub-staging (don't skip)
- **2.0** integer-membrane torch reference + S3 re-validation (§2). *No Triton.*
- **2.1** fused recurrence kernel, torch-supplied noise, bit-exact vs 2.0 (§9.2).
- **2.2** in-SRAM Philox noise regen, forward bit-exact (§9.3).
- **2.3** update kernel + end-to-end + benchmark (§9.4–9.5).
- **2.4** (optional) quantize the parallel projections (int8 GEMM + GELU LUT), RMSNorm.

## 11. Open questions / risks
- **Integer `θ` and the `s_t` mapping** may need re-tuning after the fp→int switch (§2.1).
- **Triton popcount / sub-byte support**: confirm what this Triton/GPU version exposes
  (`tl.dot(int8)` is safe; native popcount and b1 may not be) — pick SWAR vs IMMA accordingly.
- **Tiny `V`** underfills tensor cores → batch P·B·K; measure whether IMMA actually beats SWAR
  at V=32/64 before committing.
- **Scale folding** (§7) must be verified equivalent, not assumed.
- Per guardrail #3/#6: keep firing in ~10–35% and the materialized reference at every step.

## 12. Code references
- Recurrence to fuse: `SpikingM2RNN.forward`, the `for t` loop (model.py).
- Ternary quantize + per-member materialization: `ternary_quantize`, `eggroll_linear_ternary`
  (eggroll.py).
- ES update / Muon: `es_update`, `newton_schulz` (eggroll.py).
- Equivalence-test pattern to extend for the integer reference + kernel: `tests/test_equivalence.py`.
- Task harness for S3/S5 validation: `tasks/state_tracking/train_sn.py` (flags: `--mac-free
  --ternary-all`, and add an `--int-membrane` / kernel toggle as you build).
