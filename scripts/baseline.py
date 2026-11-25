"""Evaluate the base model on every fact; keep the facts it knows in data/known_ids.txt.
Cost: about 190 logprob calls + 210 short generations, under $0.01."""
import _cli  # noqa: F401

from kr import store
from kr.budget import cost
from kr.config import BASE_MODEL, PRICES
from kr.evaluate import evaluate, known_ids
from kr.tinker_model import TinkerModel


def main():
    facts, probes = store.facts(known_only=False), store.probes()
    model = TinkerModel.open(BASE_MODEL, rank=1)
    report = evaluate(model, facts, probes)
    known = known_ids(facts, report)
    store.KNOWN.write_text("\n".join(sorted(known)) + "\n")
    log = store.RunLog("baseline", {"model": BASE_MODEL})
    log.write({"step": 0, "usd": cost(model.usage, PRICES), "eval": report})
    for f in facts:
        if f.id not in known:
            print(f"  not known: {f.group:<11} {f.prompt!r} -> {report['generations'][f.id]!r}")
    for g in ("forget", "retain_near", "retain_far"):
        print(f"{g}: {sum(f.group == g and f.id in known for f in facts)} of {sum(f.group == g for f in facts)} known")
    print(f"probes hit rate {report['probes']['all']:.2f}; cost ${cost(model.usage, PRICES):.4f}; log {log.path}")


if __name__ == "__main__":
    main()
