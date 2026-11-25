"""Domain values: facts, probes and training examples. Pure: text in, immutable values out."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal, Sequence

Group = Literal["forget", "retain_near", "retain_far"]
Role = Literal["forget", "retain"]
Method = Literal["ce", "grad_diff", "npo", "simnpo"]
GROUPS: tuple[Group, ...] = ("forget", "retain_near", "retain_far")


@dataclass(frozen=True)
class Fact:
    """One prompt -> answer fact. `key` is the substring that proves the model still knows it."""
    id: str
    group: Group
    entity: str
    prompt: str
    answer: str
    key: str
    split: str  # "A" or "B", by entity within its group


@dataclass(frozen=True)
class Probe:
    """An indirect question. key == "*" means any forget-set key is a hit."""
    kind: str
    prompt: str
    key: str


@dataclass(frozen=True)
class Example:
    """One training sequence: the loss acts on `completion` given `prompt`.
    ref_logprob is the base model's log p(completion | prompt); only NPO needs it."""
    prompt: str
    completion: str
    role: Role
    ref_logprob: float | None = None


@dataclass(frozen=True)
class Loss:
    """Which unlearning objective to use. "ce" is plain cross-entropy on the given completions
    (relabel and relearning); the others push the forget completions down."""
    method: Method
    beta: float = 0.1          # NPO / SimNPO temperature
    gamma: float = 0.0         # SimNPO margin
    retain_weight: float = 1.0

    @property
    def needs_ref(self) -> bool:
        return self.method == "npo"


def _rows(text: str, width: int) -> list[list[str]]:
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) != width:
            raise ValueError(f"line {n}: expected {width} tab-separated cells, got {len(cells)}")
        rows.append([c.replace("\\n", "\n") for c in cells])
    return rows


def parse_facts(text: str) -> tuple[Fact, ...]:
    """Parse facts.tsv. Within each group, entities alternate into splits A and B in first-seen
    order, so all facts of one entity sit in one split. Forget: A is relearned, B is held out (exp 2).
    Retain: A is trained on, B is held out to measure utility honestly."""
    split_of: dict[tuple[str, str], str] = {}
    facts = []
    for i, (group, entity, prompt, answer, key) in enumerate(_rows(text, 5)):
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}")
        if not answer.startswith(" "):
            raise ValueError(f"answer must start with a space: {answer!r}")
        if key.casefold() not in answer.casefold():
            raise ValueError(f"key {key!r} not in answer {answer!r}")
        if (group, entity) not in split_of:
            split_of[(group, entity)] = "AB"[sum(g == group for g, _ in split_of) % 2]
        facts.append(Fact(f"f{i:03d}", group, entity, prompt, answer, key, split_of[(group, entity)]))
    return tuple(facts)


def parse_probes(text: str) -> tuple[Probe, ...]:
    return tuple(Probe(kind, prompt, key) for kind, prompt, key in _rows(text, 3))


def parse_pairs(text: str) -> tuple[tuple[str, str], ...]:
    return tuple((p, c) for p, c in _rows(text, 2))


def select(facts: Sequence[Fact], group: Group | None = None, split: str | None = None,
           ids: set[str] | None = None) -> tuple[Fact, ...]:
    return tuple(f for f in facts
                 if (group is None or f.group == group)
                 and (split is None or f.split == split)
                 and (ids is None or f.id in ids))


def forget_keys(facts: Sequence[Fact]) -> frozenset[str]:
    return frozenset(f.key for f in facts if f.group == "forget")


def distractors(facts: Sequence[Fact], k: int = 3) -> dict[str, tuple[str, ...]]:
    """For multiple choice: up to k answers of other entities in the same group, fixed per fact id
    (Random(str) is seeded by a hash of the string, stable across processes)."""
    out = {}
    for f in facts:
        pool = sorted({g.answer for g in facts if g.group == f.group and g.key != f.key})
        out[f.id] = tuple(Random(f.id).sample(pool, min(k, len(pool))))
    return out
