"""Experiment 2: try to recover knowledge removed by an experiment-1 run.

Attacks (plain cross-entropy fine-tuning of the unlearned LoRA):
  relearn_A  train on the true answers of forget split A only; recovery on split B means the
             knowledge of B was hidden, not removed (Hu et al. 2024; Lucki et al. 2024).
  benign     train on general music text that names no forget-set composer (data/benign.tsv).
Prompt-based attacks (QA, other languages, few-shot, lists) are the probes, scored at every eval.
"""
import argparse
from pathlib import Path

import _cli
from kr import store
from kr.budget import cost
from kr.config import BASE_MODEL, PRICES
from kr.evaluate import evaluate
from kr.items import Loss, select
from kr.tinker_model import TinkerModel
from kr.train import Schedule, retain_examples, run

CE = Loss("ce")


def attack_data(attack: str, facts) -> list[tuple[str, str]]:
    if attack == "relearn_A":
        return [(f.prompt, f.answer) for f in select(facts, "forget", "A")]
    if attack == "benign":
        return list(store.benign())
    raise ValueError(attack)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", type=Path, required=True, help="an exp1 runs/*.jsonl")
    ap.add_argument("--attack", choices=["relearn_A", "benign"], required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-usd", type=float, default=0.5)
    a = ap.parse_args()

    header, records = store.read_run(a.from_run)
    checkpoint = next(r["checkpoint"] for r in records if "checkpoint" in r)
    facts, probes = store.facts(), store.probes()
    model = TinkerModel.open(BASE_MODEL, checkpoint=checkpoint)
    schedule = Schedule(steps=a.steps, lr=a.lr, batch_size=a.batch, eval_every=a.eval_every, seed=a.seed)
    name = f"exp2_{header['args']['method']}_{a.attack}_seed{a.seed}"
    log = store.RunLog(name, {"model": BASE_MODEL, "args": {k: str(v) for k, v in vars(a).items()},
                              "checkpoint": checkpoint, "unlearn": header})

    def record(r):
        log.write(r)
        print(_cli.summary(r), flush=True)

    run(model, [], retain_examples(attack_data(a.attack, facts)), CE, schedule,
        lambda: evaluate(model, facts, probes), PRICES, a.limit_usd, record)
    log.write({"usd": cost(model.usage, PRICES)})
    print(f"log {log.path}")


if __name__ == "__main__":
    main()
