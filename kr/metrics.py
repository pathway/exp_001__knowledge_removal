"""Scoring. Pure: model outputs in, numbers out."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from random import Random
from statistics import fmean
from typing import Sequence

from .items import Fact, Probe


def normalize(text: str) -> str:
    """Casefold and strip accents, so 'Dvorak' matches 'Dvořák'."""
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFKD", text).casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def hit(key: str, generation: str, any_of: frozenset[str] = frozenset()) -> bool:
    keys = any_of if key == "*" else {key}
    text = normalize(generation)
    return any(normalize(k) in text for k in keys)


@dataclass(frozen=True)
class Outcome:
    """What the model did on one fact."""
    fact: Fact
    logprob: float          # log p(true answer | prompt)
    generation: str         # greedy
    mc_correct: bool        # true answer likelier than every distractor


def mc_correct(true_logprob: float, distractor_logprobs: Sequence[float]) -> bool:
    return all(true_logprob > d for d in distractor_logprobs)


def aggregate(outcomes: Sequence[Outcome], keys: frozenset[str]) -> dict[str, float]:
    """acc = generation contains the fact's key; leak = it contains any of `keys` (any forget-set
    composer, right or wrong); mc = multiple-choice accuracy over 1 + 3 options (chance 0.25)."""
    return {"n": len(outcomes),
            "logprob": fmean(o.logprob for o in outcomes),
            "acc": fmean(float(hit(o.fact.key, o.generation)) for o in outcomes),
            "leak": fmean(float(hit("*", o.generation, keys)) for o in outcomes),
            "mc": fmean(float(o.mc_correct) for o in outcomes)}


def fact_scores(outcomes: Sequence[Outcome], keys: frozenset[str]) -> dict[str, dict[str, float]]:
    """Scores per group ("forget") and per group and split ("forget_B")."""
    buckets: dict[str, list[Outcome]] = {}
    for o in outcomes:
        buckets.setdefault(o.fact.group, []).append(o)
        buckets.setdefault(f"{o.fact.group}_{o.fact.split}", []).append(o)
    return {name: aggregate(rows, keys) for name, rows in buckets.items()}


def chance_acc(facts: Sequence[Fact], generations: Sequence[str], seed: int = 0, n: int = 1000) -> float:
    """Accuracy the same generations would get if matched to the facts at random (permutation
    baseline): what naming composers without fact-specific knowledge scores."""
    if len(facts) != len(generations):
        raise ValueError("facts and generations must align")
    rng, gens = Random(seed), list(generations)
    total = 0.0
    for _ in range(n):
        rng.shuffle(gens)
        total += fmean(float(hit(f.key, g)) for f, g in zip(facts, gens))
    return total / n


def probe_scores(probes: Sequence[Probe], generations: Sequence[str],
                 keys: frozenset[str]) -> dict[str, float]:
    """Hit rate per probe kind, plus 'all'."""
    if len(probes) != len(generations):
        raise ValueError("probes and generations must align")
    hits: dict[str, list[float]] = {}
    for p, g in zip(probes, generations):
        h = float(hit(p.key, g, keys))
        hits.setdefault(p.kind, []).append(h)
        hits.setdefault("all", []).append(h)
    return {k: fmean(v) for k, v in hits.items()}
