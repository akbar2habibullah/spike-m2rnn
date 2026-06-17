"""
SPIKE-M2RNN: a spiking, (eventually) ternary, matrix-valued-state RNN trained
gradient-free with EGGROLL evolution strategies.

Public surface mirrors the original single-file Stage_0 script, now factored into
config / eggroll / model / data / train. `Stage_0.py` is kept FROZEN as the
numerical correctness reference (see tests/test_equivalence.py).
"""

from . import config
from .config import Config, DEFAULT
from .eggroll import (
    eggroll_linear,
    eggroll_ln,
    es_update,
    fitness_from_loss,
    newton_schulz,
    per_member_loss,
    sample_noise,
    zero_noise,
)
from .model import SpikingM2RNN

__all__ = [
    "config",
    "Config",
    "DEFAULT",
    "SpikingM2RNN",
    "eggroll_linear",
    "eggroll_ln",
    "sample_noise",
    "zero_noise",
    "per_member_loss",
    "fitness_from_loss",
    "es_update",
    "newton_schulz",
]
