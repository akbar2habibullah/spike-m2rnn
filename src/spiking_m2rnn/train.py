"""
Train / eval / generate for SPIKE-M2RNN (Stage 0).

Gradient-free throughout: the model runs under `no_grad`, params are
`requires_grad_(False)`, and the ES update is an explicit in-place step. Never add
a `.backward()`.

Run:
    python -m spiking_m2rnn.train        # from the `src/` directory
Defaults reproduce the Stage-0 Shakespeare run in `logs/Stage_0.log`.
"""

import math

import torch
import torch.nn.functional as F

from . import config
from .data import get_batch, load_data
from .eggroll import es_update, fitness_from_loss, per_member_loss
from .model import SpikingM2RNN


@torch.no_grad()
def evaluate(model, va, vocab, step, train_loss, cfg=config.DEFAULT):
    zn = model.zero_noise(cfg.device, cfg.dtype)
    losses, fire = [], None
    for _ in range(20):
        x, y = get_batch(va, cfg.batch_size, cfg.block, cfg.device)
        out = model(x, zn, 0.0, return_fire=(fire is None and model.mode == "spike"))
        logits, fire = out if isinstance(out, tuple) else (out, fire)
        losses.append(F.cross_entropy(logits[0].reshape(-1, vocab).float(), y.reshape(-1)).item())
    vloss = sum(losses) / len(losses)
    extra = f" | fire {fire:.3f}" if fire is not None else ""
    print(f"step {step:05d} | train(min) {train_loss.min().item():.3f} "
          f"| val {vloss:.3f} ({vloss / math.log(2):.2f} bpc){extra}")


@torch.no_grad()
def generate(model, stoi, itos, prompt="\n", n=400, cfg=config.DEFAULT):
    zn  = model.zero_noise(cfg.device, cfg.dtype)
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=cfg.device)
    for _ in range(n):
        logits = model(idx[:, -cfg.block:], zn, 0.0)             # (1,1,T,vocab)
        probs  = F.softmax(logits[0, :, -1, :].float(), dim=-1)  # (1,vocab)
        idx    = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())


def train(steps=3000, eval_every=100, data_path="input.txt", cfg=config.DEFAULT, compile=True):
    tr, va, vocab, stoi, itos = load_data(data_path)
    model = SpikingM2RNN(vocab, dim=cfg.dim, depth=cfg.depth, k=cfg.k_dim, v=cfg.v_dim,
                         mlp=cfg.mlp_dim, mode=cfg.mode, threshold=cfg.threshold,
                         decay=cfg.decay).to(cfg.device).to(cfg.dtype)
    model.eval(); model.requires_grad_(False)
    if compile:
        model = torch.compile(model)
    coeff   = cfg.coeff
    nparams = sum(p.numel() for p in model.P.values())
    print(f"mode={cfg.mode} params={nparams:,} pop={cfg.pop_size} sigma={cfg.sigma} "
          f"block={cfg.block} dim={cfg.dim} K={cfg.k_dim} V={cfg.v_dim} device={cfg.device}")

    for step in range(1, steps + 1):
        x, y  = get_batch(tr, cfg.batch_size, cfg.block, cfg.device)
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
            evaluate(model, va, vocab, step, loss, cfg)

    print("\n--- sample ---")
    print(generate(model, stoi, itos, cfg=cfg))


def _cli():
    import argparse
    import dataclasses

    ap = argparse.ArgumentParser(description="SPIKE-M2RNN trainer (Stage 0 char-LM).")
    ap.add_argument("--mode", choices=["spike", "tanh"], default=config.MODE,
                    help="spike = the idea; tanh = analog M2RNN baseline/control")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--data", default="input.txt", help="path to tiny-shakespeare input.txt")
    ap.add_argument("--pop", type=int, default=config.POP_SIZE)
    ap.add_argument("--sigma", type=float, default=config.SIGMA)
    ap.add_argument("--block", type=int, default=config.BLOCK)
    ap.add_argument("--chunk", type=int, default=config.CHUNK, help="slice the population (OOM relief)")
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    args = ap.parse_args()

    cfg = dataclasses.replace(config.DEFAULT, mode=args.mode, pop_size=args.pop,
                              sigma=args.sigma, block=args.block, chunk=args.chunk)
    train(steps=args.steps, eval_every=args.eval_every, data_path=args.data,
          cfg=cfg, compile=not args.no_compile)


if __name__ == "__main__":
    _cli()
