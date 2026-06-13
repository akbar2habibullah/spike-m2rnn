# State tracking — S3 / S5 word problem (Stage 0.5b) — the REAL validation

This is the **scientifically decisive** experiment (CLAUDE.md guardrail #7, DESIGN
§5). Shakespeare only earns the right to run it.

## The task
Compose a stream of permutations and predict the running product (see `s_n.py`):
- vocab = `n!`; `x[b,t]` = generator token, `y[b,t]` = cumulative product `g_1·…·g_t`.
- **S5 is non-solvable ⇒ its word problem is NC1-complete.** Transformers and
  diagonal/positive-eigenvalue linear RNNs are stuck in TC0 and *cannot* track it;
  a finite-precision non-linear RNN (FSA) can. S3 is a solvable warm-up.

`s_n.py` is the generator (self-checked); `train_sn.py` is the trainer with a
length-sweep accuracy eval. Both reuse the model + EGGROLL machinery verbatim.

## Launch
```bash
# from the repo root
python tasks/state_tracking/s_n.py                          # generator self-check
python tasks/state_tracking/train_sn.py --n 5 --mode tanh   # CONTROL first
python tasks/state_tracking/train_sn.py --n 5 --mode spike --decay 1.0
```
Key knobs: `--train-len 32 --eval-lens 16 32 64 128 256` (the generalization sweep),
`--pop --sigma --decay --threshold`, model size `--dim --depth --k --v --mlp`,
`--chunk` (OOM relief), `--no-compile`. S3 (`--n 3`, solvable warm-up) vs S5
(`--n 5`, NC1-complete).

## What "passing" means: length generalization
**Train at short T, eval at longer T.** A true FSA holds accuracy ~flat as T grows;
a positional-shortcut learner collapses past the training length. The eval prints
per-position token accuracy at each `--eval-lens`. Reproduce M2RNN's
perfect-generalization plot — this is what the spiking thesis must clear.

## Method notes (baked into `train_sn.py` defaults)
1. **Run `--mode tanh` first** as the control — its learnable forget gate can hold
   state perfectly. If tanh generalizes and spike doesn't, the culprit is the
   threshold/dead-zone (DESIGN §6.2), not ES.
2. **`decay` defaults to 1.0 here** (non-leaky integrate-and-fire). The char-LM's
   0.9 (~7-step half-life) is **fatal** for long-range tracking (DESIGN §6.4); the
   input-dependent shift-decay is the eventual MAC-free upgrade. Only affects spike.
3. Likely need larger `--pop` (chunked) and/or a **smaller model** — 224k params at
   POP=512 is variance-heavy (DESIGN §8). The trainer exposes the size knobs for this.
