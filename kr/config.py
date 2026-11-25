"""Experiment constants. One place to change model, prices and paths."""
from pathlib import Path

from .budget import QWEN3_8B

BASE_MODEL = "Qwen/Qwen3-8B"
PRICES = QWEN3_8B
ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data", ROOT / "runs"

# Qwen3-8B is chat-tuned: a bare sentence often gets a question back ("...composed by which composer?").
# Two completed non-music sentences put it in completion mode. Applied to every fact, in training and eval.
FRAME = ("The telephone was invented by Alexander Graham Bell.\n"
         "The Eiffel Tower was designed by Gustave Eiffel.\n")
