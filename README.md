# SPIKE-M2RNN — experiment launch guide

A **spiking, ternary (BitNet), matrix-valued-state RNN** — a spiking take on
**M2RNN** — trained **gradient-free with EGGROLL evolution strategies**. The goal is
**state-tracking expressivity on a multiply-free substrate**.

- **Why it's shaped this way:** [docs/DESIGN.md](docs/DESIGN.md) (source of truth for intent).
- **How we work / guardrails:** [CLAUDE.md](CLAUDE.md).
- This file = **how to launch each stage's experiment**. Climb the ladder in order;
  do **not** skip a rung (each must validate against the one below it).

## Setup
```bash
pip install torch                          # + torchvision only for the MNIST reference
# tiny-shakespeare data: put input.txt at the path you pass with --data
```
All training is gradient-free (no `.backward()`): the model runs under `no_grad`,
params are frozen, the ES update is an explicit in-place step. GPU is recommended;
the modular code also runs on CPU (small configs) for tests.

> Run trainers **from the `src/` directory** so `python -m spiking_m2rnn.<...>` resolves.

---

## Stage 0 — pipeline smoke test (char Shakespeare) — ✅ DONE / converging
**Tests:** does a spiking matrix-RNN learn end-to-end under ES at all? (Not state-tracking.)

```bash
cd src
python -m spiking_m2rnn.train --data ../path/to/input.txt
```
Common knobs: `--steps 3000 --eval-every 100 --pop 512 --sigma 0.05 --block 64`,
`--chunk 128` (population slicing for OOM), `--no-compile` (skip `torch.compile`).

**Expect** (see [logs/Stage_0.log](logs/Stage_0.log)): val 5.90 → ~4.23 bpc, firing
self-stabilizing **~33%**. It will **not** reach Adam nanoGPT (~1.4 bpc) — expected for
ES at this scale. Watch the `fire` readout: silent/saturated = no signal (keep ~10–35%).

---

## Stage 0.5 — the real validation
### (a) `tanh` baseline control — ✅ runnable now
**Tests:** isolates the spiking nonlinearity from ES. Recovers analog M2RNN
(`Z=tanh(HW+kvᵀ)`). If tanh learns and spike doesn't, the culprit is the
threshold/dead-zone (DESIGN §6.2), **not** ES.
```bash
cd src
python -m spiking_m2rnn.train --mode tanh --data ../path/to/input.txt
```

### (b) S3/S5 state-tracking with length generalization — 🚧 generator done, trainer TODO
**Tests:** the architecture's actual reason for being. Predict the running product of
a permutation stream; **train at short T, eval at longer T**. S5 is non-solvable ⇒
NC1-complete (a Transformer/diagonal-SSM cannot track it; a finite-precision
non-linear RNN can).

The data generator exists and is self-checked:
```bash
python tasks/state_tracking/s_n.py        # S3 (vocab 6) + S5 (vocab 120) self-check
```
The trainer (length-sweep **accuracy** eval, not bpc) is not built yet — see
[tasks/state_tracking/README.md](tasks/state_tracking/README.md). When standing it
up: run the `tanh` control first, and replace `DECAY=0.9` with the input-dependent
shift-decay (DESIGN §6.4) — a 7-step half-life erases early state.

---

## Stage 1a — MAC-free conversion — 🚧 TODO
Integer membrane, shift-based leak `U − (U≫n)`, subtractive reset, input-dependent
shift-decay. **Must numerically validate against Stage 0** before proceeding. Isolates
the multiply-free claim. (No launch command yet.)

## Stage 1b — ternary `W` (BitNet) — 🚧 TODO
Float latent master + quantize-in-forward + **per-member 2-bit materialization**
(breaks EGGROLL's no-materialize trick — DESIGN §6.6); σ tuned to bin width. **Must
validate against Stage 0/1a.** Isolates the BitNet cost. (No launch command yet.)

## Stage 2 — Triton kernel — 🚧 TODO
Bit-packed ternary × binary spikes via AND+popcount, **in-SRAM Philox noise**, fused
recurrence. **Must be bit-exact vs Stage 1** (kernel invariants: DESIGN §7). (No
launch command yet.)

---

## Reference experiment (not a stage) — EGGROLL ViT on MNIST
The canonical ES machinery this project mirrors; confirms EGGROLL converges.
```bash
python reference/eggroll_vit_mnist.py     # ~81% in 5 epochs; see logs/eggroll_vit_mnist.log
```

## Tests — cross-stage numerical equivalence
The refactor guard: the modular package must be **bit-identical** to the frozen
single-file `Stage_0.py` reference (preserve this path at every stage — guardrail #2).
```bash
python -m pytest tests/ -q               # or: python tests/test_equivalence.py
```
