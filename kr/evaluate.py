"""Evaluate a model on facts and probes through the port."""
from __future__ import annotations

from typing import Sequence

from .items import Fact, Probe, distractors, forget_keys
from .metrics import Outcome, fact_scores, hit, mc_correct, probe_scores
from .port import Model

MAX_TOKENS = 12


def evaluate(model: Model, facts: Sequence[Fact], probes: Sequence[Probe]) -> dict:
    """Scores per group and split, per probe kind; plus every generation for audit.
    One logprob batch scores the true answers and the multiple-choice distractors together."""
    options = distractors(facts)
    pairs = [(f.prompt, f.answer) for f in facts] + [(f.prompt, d) for f in facts for d in options[f.id]]
    lps = model.answer_logprobs(pairs)
    true_lps, rest = lps[:len(facts)], iter(lps[len(facts):])
    wrong = [[next(rest) for _ in options[f.id]] for f in facts]
    gens = model.generate([f.prompt for f in facts] + [p.prompt for p in probes], MAX_TOKENS)
    fact_gens, probe_gens = gens[:len(facts)], gens[len(facts):]
    outcomes = [Outcome(f, lp, g, mc_correct(lp, w)) for f, lp, g, w in zip(facts, true_lps, fact_gens, wrong)]
    keys = forget_keys(facts)
    report = fact_scores(outcomes, keys)
    report["probes"] = probe_scores(probes, probe_gens, keys)
    report["generations"] = {f.id: g for f, g in zip(facts, fact_gens)} | {
        f"probe{i:02d}": g for i, g in enumerate(probe_gens)}
    return report


def known_ids(facts: Sequence[Fact], report: dict) -> set[str]:
    """Facts the model answers correctly (key in greedy generation): the only ones worth forgetting
    or retaining, since a fact the base model never knew gives no signal."""
    return {f.id for f in facts if hit(f.key, report["generations"][f.id])}

