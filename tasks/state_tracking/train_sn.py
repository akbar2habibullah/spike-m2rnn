"""
Stage 0.5b -- S3/S5 state-tracking trainer with LENGTH GENERALIZATION eval.

The decisive experiment (DESIGN 5, CLAUDE.md guardrail #7). Train on short
sequences, evaluate on longer ones, and watch whether accuracy holds. A true FSA
(finite-precision non-linear RNN) holds flat as length grows; a TC0 shortcut
learner collapses past the training length.

Reuses the EGGROLL ES machinery and the model verbatim; only the data source
(permutation streams from `s_n.py`) and the eval metric (per-position token
accuracy, swept over length) differ from the char-LM trainer.

Run (from the repo root):
    python tasks/state_tracking/train_sn.py --n 5 --mode tanh
    python tasks/state_tracking/train_sn.py --n 5 --mode spike --decay 1.0

Method notes baked into the defaults:
  * Run `--mode tanh` FIRST as the control (its learnable forget gate can hold
    state perfectly). If tanh generalizes and spike doesn't, the culprit is the
    threshold/dead-zone (DESIGN 6.2), not ES.
  * `decay` defaults to 1.0 here (non-leaky integrate-and-fire) -- the char-LM's
    0.9 (~7-step half-life) is FATAL for long-range tracking (DESIGN 6.4). Only
    affects spike mode (tanh uses its forget gate).
  * NoPE + a sequential recurrence => the model runs at ANY length, so eval at
    lengths far past training is well-defined. torch.compile recompiles per length
    (benign).
"""

import argparse
import dataclasses
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from spiking_m2rnn import config                                  # noqa: E402
from spiking_m2rnn.eggroll import es_update, fitness_from_loss, per_member_loss  # noqa: E402
from spiking_m2rnn.model import SpikingM2RNN                      # noqa: E402
from s_n import SymmetricGroup, make_batch                        # noqa: E402


@torch.no_grad()
def eval_length_sweep(model, group, lengths, cfg, batches=20):
    """Per-position token accuracy at each eval length (mean over `batches`)."""
    zn = model.zero_noise(cfg.device, cfg.dtype)
    acc = {}
    for L in lengths:
        correct = total = 0
        for _ in range(batches):
            x, y = make_batch(group, cfg.batch_size, L, cfg.device)
            logits = model(x, zn, 0.0)[0]                         # (B,T,vocab)
            pred = logits.argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
        acc[L] = correct / total
    return acc


def _fmt(acc):
    return "  ".join(f"L{L}:{a*100:5.1f}%" for L, a in acc.items())


def train(group, train_len, eval_lens, steps, eval_every, cfg, compile=True):
    vocab = group.size
    model = SpikingM2RNN(vocab, dim=cfg.dim, depth=cfg.depth, k=cfg.k_dim, v=cfg.v_dim,
                         mlp=cfg.mlp_dim, mode=cfg.mode, threshold=cfg.threshold,
                         decay=cfg.decay).to(cfg.device).to(cfg.dtype)
    model.eval(); model.requires_grad_(False)
    if compile:
        model = torch.compile(model)
    coeff = cfg.coeff
    nparams = sum(p.numel() for p in model.P.values())
    print(f"task=S{group.n} vocab={vocab} mode={cfg.mode} params={nparams:,} "
          f"pop={cfg.pop_size} sigma={cfg.sigma} decay={cfg.decay} "
          f"train_len={train_len} eval_lens={eval_lens} device={cfg.device}")
    chance = 1.0 / vocab
    print(f"(chance accuracy = {chance*100:.2f}%)")

    for step in range(1, steps + 1):
        x, y = make_batch(group, cfg.batch_size, train_len, cfg.device)
        noise = model.sample_noise(cfg.pop_size, cfg.rank, cfg.device, cfg.dtype)
        with torch.no_grad():
            if cfg.chunk is None:
                loss = per_member_loss(model(x, noise, cfg.sigma), y)
            else:
                parts = []
                for s in range(0, cfg.pop_size, cfg.chunk):
                    sub = {k: (tuple(t[s:s + cfg.chunk] for t in vv) if isinstance(vv, tuple) else vv[s:s + cfg.chunk])
                           for k, vv in noise.items()}
                    parts.append(per_member_loss(model(x, sub, cfg.sigma), y))
                loss = torch.cat(parts)
        fit = fitness_from_loss(loss).to(cfg.dtype)
        es_update(model.P, noise, fit, coeff, cfg.rank_scale)

        if step == 1 or step % eval_every == 0:
            acc = eval_length_sweep(model, group, eval_lens, cfg)
            print(f"step {step:05d} | train(min loss) {loss.min().item():.3f} | {_fmt(acc)}")


def _cli():
    ap = argparse.ArgumentParser(description="S_n state-tracking trainer (Stage 0.5b).")
    ap.add_argument("--n", type=int, default=5, help="symmetric group S_n (3 or 5)")
    ap.add_argument("--mode", choices=["spike", "tanh"], default="tanh",
                    help="run tanh control FIRST; then spike")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--train-len", type=int, default=32)
    ap.add_argument("--eval-lens", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--pop", type=int, default=config.POP_SIZE)
    ap.add_argument("--sigma", type=float, default=config.SIGMA)
    ap.add_argument("--decay", type=float, default=1.0,
                    help="spike membrane leak; 1.0 = non-leaky IF (default for tracking)")
    ap.add_argument("--threshold", type=float, default=config.THRESHOLD)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--chunk", type=int, default=config.CHUNK)
    # model size (DESIGN 8: a smaller model may give cleaner ES signal)
    ap.add_argument("--dim", type=int, default=config.DIM)
    ap.add_argument("--depth", type=int, default=config.DEPTH)
    ap.add_argument("--k", type=int, default=config.K_DIM)
    ap.add_argument("--v", type=int, default=config.V_DIM)
    ap.add_argument("--mlp", type=int, default=config.MLP_DIM)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    cfg = dataclasses.replace(
        config.DEFAULT, mode=args.mode, pop_size=args.pop, sigma=args.sigma,
        decay=args.decay, threshold=args.threshold, batch_size=args.batch, chunk=args.chunk,
        dim=args.dim, depth=args.depth, k_dim=args.k, v_dim=args.v, mlp_dim=args.mlp,
    )
    group = SymmetricGroup(args.n)
    train(group, args.train_len, args.eval_lens, args.steps, args.eval_every,
          cfg, compile=not args.no_compile)


if __name__ == "__main__":
    _cli()
