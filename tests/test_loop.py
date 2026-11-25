"""Training loop and evaluation through the fake: fast, deterministic, no network."""
import pytest

from kr.budget import OverBudget, Prices
from kr.evaluate import evaluate, known_ids
from kr.fake_model import FakeModel
from kr.items import Loss, parse_facts, parse_probes, select
from kr.train import SUBSTITUTE, Schedule, forget_examples, retain_examples, run

FACTS = parse_facts(
    "forget\tMozart\tThe Magic Flute is by\t Mozart\tMozart\n"
    "forget\tBach\tThe Brandenburg Concertos are by\t Bach\tBach\n"
    "retain_far\tP\tThe capital of France is\t Paris\tParis\n"
    "forget\tUnknownToModel\tObscure piece by\t Nobody\tNobody\n")
PROBES = parse_probes("list\tFamous composers:\t*")
PRICES = Prices(1.0, 1.0, 1.0)


def base_model():
    return FakeModel({f.prompt: f.answer for f in FACTS if f.entity != "UnknownToModel"}
                     | {"Famous composers:": " Mozart and Bach"})


def test_known_ids_drops_facts_the_base_model_never_knew():
    report = evaluate(base_model(), FACTS, PROBES)
    assert known_ids(FACTS, report) == {"f000", "f001", "f002"}
    assert report["probes"]["all"] == 1.0


@pytest.mark.parametrize("method", ["ce", "grad_diff", "npo"])
def test_unlearning_removes_forget_and_keeps_retain(method):
    model = base_model()
    facts = select(FACTS, ids={"f000", "f001", "f002"})
    forget, retain = select(facts, "forget"), select(facts, "retain_far")
    ref = model.answer_logprobs([(f.prompt, f.answer) for f in forget]) if method == "npo" else None
    loss = Loss(method)
    history = run(model, forget_examples(forget, loss, ref), retain_examples([(f.prompt, f.answer) for f in retain]),
                  loss, Schedule(steps=2, lr=1e-4, eval_every=1), lambda: evaluate(model, facts, PROBES),
                  PRICES, limit_usd=1.0)
    first, last = history[0]["eval"], history[-1]["eval"]
    assert first["forget"]["acc"] == 1.0 and last["forget"]["acc"] == 0.0
    assert last["retain_far"]["acc"] == 1.0
    assert [h["step"] for h in history] == [0, 1, 2]
    assert history[-1]["usd"] >= history[0]["usd"]


def test_relabel_uses_substitute_and_npo_needs_ref():
    forget = select(FACTS, "forget")
    assert {e.completion for e in forget_examples(forget, Loss("ce"))} == {SUBSTITUTE}
    with pytest.raises(ValueError, match="reference"):
        forget_examples(forget, Loss("npo"))


def test_budget_stops_before_overspending():
    model = base_model()
    ex = retain_examples([("The capital of France is", " Paris")])
    with pytest.raises(OverBudget):
        run(model, [], ex, Loss("ce"), Schedule(steps=5, lr=1e-4), lambda: {}, PRICES, limit_usd=1e-9)


def test_checkpoint_round_trip_supports_relearning_from_unlearned_weights():
    model = base_model()
    model.train_step(forget_examples(select(FACTS, "forget"), Loss("grad_diff")), Loss("grad_diff"), 1e-4)
    restored = FakeModel.from_checkpoint(model.save("unlearned"))
    assert restored.generate(["The Magic Flute is by"], 4) == model.generate(["The Magic Flute is by"], 4)
