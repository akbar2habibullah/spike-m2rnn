# CLAUDE.md — SPIKE-M2RNN

Operating guide for coding sessions. Keep this short. Deep rationale lives in
`docs/DESIGN.md` — **read it before changing the architecture or starting a new
stage.**

## What this project is
A **spiking, ternary (BitNet), matrix-valued-state RNN** — a spiking take on
**M2RNN** — trained **gradient-free with EGGROLL evolution strategies**. ES is the
keystone: it removes surrogate gradients, the straight-through estimator, and BPTT
all at once. The goal is **state-tracking expressivity on a multiply-free substrate**.

## Status
**Stage 0 converges on char Shakespeare** (GPU, fp16): val 5.90 → ~4.23 bpc, firing
~33%. This validates the *pipeline*, not state-tracking. See `docs/DESIGN.md §4–5`.

## Repository layout (target)
```
.
├── CLAUDE.md
├── docs/DESIGN.md            # full context — the source of truth for design intent
├── reference/
│   └── eggroll_vit_mnist.py  # working EGGROLL ViT; canonical ES machinery to mirror
├── src/spiking_m2rnn/
│   ├── config.py            # constants + Config dataclass (per-task overrides)
│   ├── eggroll.py           # eggroll_linear, eggroll_ln, sample_noise, fitness, es_update
│   ├── model.py             # SpikingM2RNN (modes: "spike" | "tanh")
│   ├── data.py              # char data + get_batch
│   ├── train.py             # train / eval / generate
│   └── Stage_0.py           # FROZEN single-file reference (do not edit; equivalence target)
├── tasks/
│   ├── shakespeare/          # char-LM smoke test (Stage 0)
│   └── state_tracking/       # S3/S5 length-gen — the REAL validation (s_n.py done; trainer TODO)
├── kernels/                  # Triton (Stage 2, TODO)
├── tests/                    # cross-stage numerical equivalence (test_equivalence.py)
└── logs/
```
The split out of the single-file `Stage_0.py` into `src/spiking_m2rnn/` is **done**
and behaviour-preserving: `tests/test_equivalence.py` asserts the modular path is
bit-identical to the frozen `Stage_0.py` (forward both modes + a full ES step, 0.0
error in float64). Keep `Stage_0.py` frozen as the reference per guardrail #2.

`.gitignore` should cover: `__pycache__/`, `*.pyc`, `data/**/*.bin`,
`data/**/input.txt`, `*.log`, `out*/`, `checkpoints/`, `*.pt`, `wandb/`,
`.ipynb_checkpoints/`.

## How to run
```bash
# data: tiny-shakespeare input.txt next to the script (or point data_path at it)
python -m spiking_m2rnn.train          # or: python Stage_0.py
```
GPU: set `DTYPE=torch.bfloat16` or `float16` (fp16 confirmed stable at Stage 0).
Config is module-level constants at the top of the model/train file.

## Conventions (match the EGGROLL ViT exactly — do not drift)
- `ParameterDict` keys use `_` not `.` (e.g. `block0_q_w`).
- Key suffix `_w` ⇒ a matrix ⇒ low-rank `(A, B)` noise + update `Σ_p f_p A_p B_pᵀ`.
  Any other key ⇒ dense noise + `tensordot` update.
- `eggroll_linear` expects `(..., S, I)` — the second-to-last dim is "S", last is
  in-features `I`. The recurrence transition passes the state `(P,B,K,V)` with `S=K`,
  `I=V`. The first layer takes 3-D `(B,T,vocab)` and broadcasts the population dim.
- Update step size `coeff = LR / (POP·SIGMA)`; `RANK_SCALE = 1/√RANK`.
- The whole thing is gradient-free: model runs under `no_grad`, params
  `requires_grad_(False)`. Never add a `.backward()`.

## Hard rules / guardrails
1. **Do not skip stages.** Stage 1 (ternary) must be **numerically validated against
   Stage 0** before Stage 2 (kernel). Stage 2 must be **bit-exact vs Stage 1**.
2. **Preserve the materialized reference path** at every stage for equivalence tests
   (the core EGGROLL math matches brute-force to ~1e-15 in float64 — keep a test).
3. **`SIGMA=0.05` is deliberate** — the hard threshold makes fitness piecewise-
   constant in most weights (the "dead zone"). If a spiking run is flat, raise
   `SIGMA` first, then `POP`. Do not "fix" this by reaching for gradients. (DESIGN §6.2)
4. **Ternary breaks the no-materialize trick** → per-member 2-bit weights + a float
   latent master. The perf model changes; low-rank's role shrinks. (DESIGN §6.6)
5. **Kernel noise invariants** (Stage 2): forward & update must regenerate
   bit-identical Gaussian noise via in-SRAM Philox keyed by `(step, p)`; bake
   antithetic pairing into the seed map; validate against a materialized torch path
   (Triton-Philox ≠ `torch.randn`). (DESIGN §7)
6. **Keep firing in ~10–35%.** Watch the eval readout; silent/saturated = no signal.
7. **Shakespeare is a smoke test, not the goal.** The decisive experiment is **S3/S5
   state-tracking with length generalization**. (DESIGN §5)

## Current next task
Stage 0.5: (a) run the `tanh` (analog M2RNN) baseline as a control on the same task;
(b) stand up the **S3/S5 state-tracking** task with train/eval at different lengths
and check length generalization. Before scaling: consider larger `POP` (chunked)
and/or a smaller model — 224k params at `POP=512` is variance-heavy.
