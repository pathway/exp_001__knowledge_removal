"""Shared CLI helpers: path setup, one-line progress print."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summary(record: dict) -> str:
    """One readable line per step: forget/retain accuracy, probe hit rate, spend."""
    parts = [f"step {record['step']:>3}"]
    if "loss" in record:
        parts.append(f"loss {record['loss']:.3f}")
    ev = record.get("eval")
    if ev:
        for g in ("forget_A", "forget_B", "retain_near_B", "retain_far_B"):
            if g in ev:
                parts.append(f"{g} acc {ev[g]['acc']:.2f} leak {ev[g]['leak']:.2f} mc {ev[g]['mc']:.2f}")
        parts.append(f"probes {ev['probes']['all']:.2f}")
    parts.append(f"${record['usd']:.4f}")
    return " | ".join(parts)
