# State tracking — S3 / S5 word problem (Stage 0.5b) — the REAL validation

This is the **scientifically decisive** experiment (CLAUDE.md guardrail #7, DESIGN
§5). Shakespeare only earns the right to run it.

## The task
Compose a stream of permutations and predict the running product (see `s_n.py`):
- vocab = `n!`; `x[b,t]` = generator token, `y[b,t]` = cumulative product `g_1·…·g_t`.
- **S5 is non-solvable ⇒ its word problem is NC1-complete.** Transformers and
  diagonal/positive-eigenvalue linear RNNs are stuck in TC0 and *cannot* track it;
  a finite-precision non-linear RNN (FSA) can. S3 is a solvable warm-up.

`s_n.py` is generator-only and self-checked. It exposes `get_batch_fn(group)` matching
`data.get_batch`'s `(data, batch, block, device)` signature, so the train loop stays
task-agnostic.

## What "passing" means: length generalization
**Train at short T, eval at longer T.** A true FSA holds accuracy ~flat as T grows;
a positional-shortcut learner collapses. Reproduce M2RNN's perfect-generalization
plot. This is what the spiking thesis must clear.

## TODO to stand this up (the actual next step)
1. A train entry point that swaps `data.get_batch` → `s_n.get_batch_fn(SymmetricGroup(n))`
   and uses an accuracy (not bpc) eval that sweeps eval length.
2. **Run the `tanh` baseline first** (`Config(mode="tanh")`) as the control — if tanh
   learns and spike doesn't, the culprit is the threshold/dead-zone (DESIGN §6.2), not ES.
3. **`DECAY=0.9` is fatal here** (~7-step half-life erases early state). For long-range
   tracking switch to the input-dependent shift-decay (DESIGN §6.4) before blaming ES.
4. Likely need larger `POP` (chunked) and/or a smaller model — 224k params at POP=512
   is variance-heavy (DESIGN §8).
