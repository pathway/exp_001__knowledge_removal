"""Unlearning losses over per-token logprobs. Pure torch: tensors in, scalar loss out.

Each sequence comes as (token_logprobs, weights, role, ref_logprob). weights mask the completion.
Every loss = forget term + retain_weight * mean retain cross-entropy (per token).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .items import Loss


@dataclass(frozen=True)
class Seq:
    logprobs: torch.Tensor     # per target token
    weights: torch.Tensor      # 1 on completion tokens, 0 elsewhere
    role: str
    ref_logprob: float | None = None

    @property
    def total(self) -> torch.Tensor:
        return (self.logprobs * self.weights).sum()

    @property
    def length(self) -> torch.Tensor:
        return self.weights.sum().clamp(min=1.0)


def _mean(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs).mean() if xs else torch.zeros(())


def forget_term(loss: Loss, seqs: Sequence[Seq]) -> torch.Tensor:
    if loss.method == "grad_diff":    # gradient ascent on forget = minimise per-token log p
        return _mean([s.total / s.length for s in seqs])
    if loss.method == "npo":          # Zhang et al. 2024: (2/b) softplus(b (log p - log p_ref))
        return _mean([2 / loss.beta * F.softplus(loss.beta * (s.total - s.ref_logprob)) for s in seqs])
    if loss.method == "simnpo":       # Fan et al. 2024: length-normalised, no reference model
        return _mean([2 / loss.beta * F.softplus(loss.beta * s.total / s.length + loss.gamma) for s in seqs])
    if loss.method == "ce":           # relabel: forget examples already carry substitute completions
        return _mean([-s.total / s.length for s in seqs])
    raise ValueError(loss.method)


def total_loss(loss: Loss, seqs: Sequence[Seq]) -> tuple[torch.Tensor, dict[str, float]]:
    if loss.needs_ref and any(s.ref_logprob is None for s in seqs if s.role == "forget"):
        raise ValueError("NPO needs ref_logprob on every forget sequence")
    forget = [s for s in seqs if s.role == "forget"]
    retain = [s for s in seqs if s.role == "retain"]
    f = forget_term(loss, forget)
    r = _mean([-s.total / s.length for s in retain])
    value = f + loss.retain_weight * r
    return value, {"forget_term": float(f.detach()), "retain_ce": float(r.detach())}
