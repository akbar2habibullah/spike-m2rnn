"""
Char-level data (nanoGPT convention: full vocab, 90/10 split).

This is the Stage-0 Shakespeare smoke-test loader. The Stage-0.5 state-tracking
task (S3/S5 permutation word problems with length generalization) gets its own
generator under `tasks/state_tracking/`; it can reuse `get_batch`'s (x, y) shape
contract -- (B, T) long tensors -- so the train loop stays task-agnostic.
"""

import torch


def load_data(path="input.txt"):
    text = open(path, "r").read()
    chars = sorted(set(text))
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for i, c in enumerate(chars)}
    ids   = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n     = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars), stoi, itos


def get_batch(data, batch, block, device):
    ix = torch.randint(len(data) - block - 1, (batch,)).tolist()
    x  = torch.stack([data[i:i + block]         for i in ix])
    y  = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x.to(device), y.to(device)
