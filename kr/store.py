"""File I/O at the edge: read the data files, write run logs. No domain logic."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA, FRAME, RUNS
from .items import Fact, Probe, parse_facts, parse_pairs, parse_probes, select

KNOWN = DATA / "known_ids.txt"   # derived by scripts/baseline.py: facts the base model knows


def facts(known_only: bool = True) -> tuple[Fact, ...]:
    all_facts = tuple(replace(f, prompt=FRAME + f.prompt) for f in parse_facts((DATA / "facts.tsv").read_text()))
    if not known_only:
        return all_facts
    if not KNOWN.exists():
        raise FileNotFoundError(f"{KNOWN} missing: run scripts/baseline.py first")
    return select(all_facts, ids=set(KNOWN.read_text().split()))


def probes() -> tuple[Probe, ...]:
    return parse_probes((DATA / "probes.tsv").read_text())


def benign() -> tuple[tuple[str, str], ...]:
    return parse_pairs((DATA / "benign.tsv").read_text())


class RunLog:
    """One JSONL file per run: a header line, then one line per step. Flushed every line."""

    def __init__(self, name: str, header: dict):
        RUNS.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = RUNS / f"{stamp}__{name}.jsonl"
        self._f = self.path.open("w")
        self.write({"header": header})

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()


def read_run(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(x) for x in Path(path).read_text().splitlines()]
    return lines[0]["header"], lines[1:]
