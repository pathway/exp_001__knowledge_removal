"""Experiment 1: unlearn every known forget fact (splits A and B) with one method.

Train on: forget A+B, retain A.  Report: forget A/B, retain B (held out), indirect probes.
Saves the final LoRA (7-day TTL) for experiment 2.
"""
import argparse

import _cli
from kr import store
from kr.budget import cost
from kr.config import BASE_MODEL, PRICES
from kr.evaluate import evaluate
from kr.items import Loss, select
from kr.tinker_model import TinkerModel
from kr.train import Schedule, forget_examples, retain_examples, run

METHODS = {  # name -> loss; "relabel" is Who's Harry Potter style CE on a substitute answer
    "relabel": Loss("ce"),
    "grad_diff": Loss("grad_diff"),
    "npo": Loss("npo", beta=0.1),
    "simnpo": Loss("simnpo", beta=2.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-usd", type=float, default=1.0)
    a = ap.parse_args()

    loss = METHODS[a.method]
    facts, probes = store.facts(), store.probes()
    forget = select(facts, "forget")
    retain = [f for f in facts if f.group != "forget" and f.split == "A"]
    model = TinkerModel.open(BASE_MODEL, rank=a.rank, seed=a.seed)
    ref = model.answer_logprobs([(f.prompt, f.answer) for f in forget]) if loss.needs_ref else None

    schedule = Schedule(steps=a.steps, lr=a.lr, batch_size=a.batch, eval_every=a.eval_every, seed=a.seed)
    log = store.RunLog(f"exp1_{a.method}_seed{a.seed}", {"model": BASE_MODEL, "args": vars(a), "loss": loss.__dict__})

    def record(r):
        log.write(r)
        print(_cli.summary(r), flush=True)

    run(model, forget_examples(forget, loss, ref), retain_examples([(f.prompt, f.answer) for f in retain]),
        loss, schedule, lambda: evaluate(model, facts, probes), PRICES, a.limit_usd, record)
    path = model.save(f"exp1-{a.method}-steps{a.steps}-seed{a.seed}")
    log.write({"checkpoint": path, "usd": cost(model.usage, PRICES)})
    print(f"checkpoint {path}\nlog {log.path}")


if __name__ == "__main__":
    main()
