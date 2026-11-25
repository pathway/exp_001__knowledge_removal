"""Token usage and cost. Pure."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prices:
    """USD per million tokens. custom_factor: forward_backward_custom does an extra forward (1.5x FLOPs)."""
    prefill: float
    sample: float
    train: float
    custom_factor: float = 1.5


# Qwen/Qwen3-8B, from the Tinker models & pricing page, read 2026-11-24.
QWEN3_8B = Prices(prefill=0.195, sample=0.60, train=0.44)


@dataclass(frozen=True)
class Usage:
    prefill: int = 0
    sample: int = 0
    train: int = 0
    custom_train: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prefill + other.prefill, self.sample + other.sample,
                     self.train + other.train, self.custom_train + other.custom_train)


def cost(usage: Usage, prices: Prices) -> float:
    per_token = 1e-6
    return per_token * (usage.prefill * prices.prefill + usage.sample * prices.sample
                        + (usage.train + prices.custom_factor * usage.custom_train) * prices.train)


class OverBudget(RuntimeError):
    pass


def check(spent: float, next_estimate: float, limit: float) -> None:
    """Fail fast before an operation that would cross the limit."""
    if spent + next_estimate > limit:
        raise OverBudget(f"spent ${spent:.4f} + next ${next_estimate:.4f} > limit ${limit:.2f}")

