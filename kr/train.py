"""Training loop over the port: unlearning (exp 1) and relearning attacks (exp 2) share it."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable, Sequence

from .budget import Prices, check, cost
from .items import Example, Fact, Loss
from .port import Model

SUBSTITUTE = " an unknown musician"   # relabel target for forget facts (Who's Harry Potter style)


@dataclass(frozen=True)
class Schedule:
    steps: int
    lr: float
    batch_size: int = 16      # per role
    eval_every: int = 10
    seed: int = 0


def forget_examples(facts: Sequence[Fact], loss: Loss, ref: Sequence[float] | None = None) -> list[Example]:
    """For "ce" (relabel) the target is the substitute; otherwise the true answer, pushed down."""
    if loss.needs_ref and (ref is None or len(ref) != len(facts)):
        raise ValueError("NPO needs one reference logprob per forget fact")
    if loss.method == "ce":
        return [Example(f.prompt, SUBSTITUTE, "forget") for f in facts]
    refs = ref if loss.needs_ref else [None] * len(facts)
    return [Example(f.prompt, f.answer, "forget", r) for f, r in zip(facts, refs)]


def retain_examples(pairs: Sequence[tuple[str, str]]) -> list[Example]:
    return [Example(p, c, "retain") for p, c in pairs]


def run(model: Model, forget: Sequence[Example], retain: Sequence[Example], loss: Loss,
        schedule: Schedule, evaluate: Callable[[], dict], prices: Prices, limit_usd: float,
        log: Callable[[dict], None] = lambda r: None) -> list[dict]:
    """Train for schedule.steps. Evaluates at step 0, every eval_every steps and at the end.
    Stops with OverBudget before any step that would cross limit_usd."""
    rng = Random(schedule.seed)
    history = [{"step": 0, "usd": cost(model.usage, prices), "eval": evaluate()}]
    log(history[0])
    for step in range(1, schedule.steps + 1):
        batch = (rng.sample(list(forget), min(schedule.batch_size, len(forget)))
                 + rng.sample(list(retain), min(schedule.batch_size, len(retain))))
        check(cost(model.usage, prices), cost(model.estimate(batch, loss), prices), limit_usd)
        record = {"step": step, **model.train_step(batch, loss, schedule.lr)}
        if step % schedule.eval_every == 0 or step == schedule.steps:
            record["eval"] = evaluate()
        record["usd"] = cost(model.usage, prices)
        log(record)
        history.append(record)
    return history
