"""
Central configuration for SPIKE-M2RNN.

These are the module-level constants the Stage-0 single-file script kept at the top
of the file; they live here now so model / data / train / eggroll all agree on one
source of truth. Defaults reproduce the Stage-0 Shakespeare run EXACTLY
(val 5.90 -> ~4.23 bpc, firing ~33%). Change them per-task, not per-module.

A `Config` dataclass is also provided so Stage 0.5 tasks (tanh baseline, S3/S5
length-generalization) can override hyperparameters -- e.g. train and eval at
different `block` lengths -- without mutating globals. `DEFAULT` mirrors the
constants below.
"""

import math
from dataclasses import dataclass

import torch

# ---- runtime ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16          # fp32 is safest for the threshold dynamics; bf16 on GPU
MODE   = "spike"                # "spike" (the idea) | "tanh" (analog M2RNN baseline)

# ---- model (deliberately SMALL: ES wants few params + large population) ----
DIM     = 128
DEPTH   = 2
K_DIM   = 64                    # key head dim
V_DIM   = 64                    # value head dim   (matrix state is K_DIM x V_DIM)
MLP_DIM = 256
BLOCK   = 64                    # context length (kept short: the recurrence is sequential)

# ---- spiking dynamics (fixed for Stage 0; make learnable / input-dependent later) ----
THRESHOLD = 1.0
DECAY     = 0.9                 # ~7-step half-life: fine for char-LM, NOT long-range state-tracking

# ---- EGGROLL / ES ----
BATCH_SIZE = 24
POP_SIZE   = 512                # raise as far as memory allows; chunk if OOM
RANK       = 1
SIGMA      = 0.05               # larger than the ViT's 0.01 ON PURPOSE -- dead-zone (DESIGN 6.2)
LR         = 0.05
CHUNK      = None               # e.g. 128 to slice the population; None = all at once
RANK_SCALE = 1.0 / math.sqrt(RANK)

torch.set_float32_matmul_precision("high")


@dataclass
class Config:
    """Per-run hyperparameters. Override fields for Stage 0.5 controls/tasks."""
    # runtime
    device: torch.device = DEVICE
    dtype: torch.dtype = DTYPE
    mode: str = MODE
    # model
    dim: int = DIM
    depth: int = DEPTH
    k_dim: int = K_DIM
    v_dim: int = V_DIM
    mlp_dim: int = MLP_DIM
    block: int = BLOCK
    # spiking dynamics
    threshold: float = THRESHOLD
    decay: float = DECAY
    # ES
    batch_size: int = BATCH_SIZE
    pop_size: int = POP_SIZE
    rank: int = RANK
    sigma: float = SIGMA
    lr: float = LR
    chunk: "int | None" = CHUNK
    # Muon-style orthogonalized update (opt-in): decouples step size from gradient-estimate
    # magnitude so POP only sharpens direction (DESIGN 6.2 dead-zone). muon_lr is the step.
    muon: bool = False
    muon_lr: float = 0.02

    @property
    def rank_scale(self) -> float:
        return 1.0 / math.sqrt(self.rank)

    @property
    def coeff(self) -> float:
        """ES update step size: LR / (POP * SIGMA)."""
        return self.lr / (self.pop_size * self.sigma)


DEFAULT = Config()
