"""Pure core: plain values in, values out. No mocks."""
from pathlib import Path

import pytest
import torch
from hypothesis import given, strategies as st

from kr import budget, losses
from kr.items import Loss, distractors, forget_keys, parse_facts, parse_probes, select
from kr.metrics import Outcome, chance_acc, fact_scores, hit, mc_correct, normalize, probe_scores

DATA = Path(__file__).resolve().parents[1] / "data"
FACTS = parse_facts((DATA / "facts.tsv").read_text())


def test_real_data_parses_and_every_entity_sits_in_one_split():
    assert len(select(FACTS, "forget")) > 100
    for f in FACTS:
        assert len({g.split for g in FACTS if (g.group, g.entity) == (f.group, f.entity)}) == 1
    a, b = select(FACTS, "forget", "A"), select(FACTS, "forget", "B")
    assert abs(len(a) - len(b)) <= 3
    assert {f.split for f in select(FACTS, "retain_far")} == {"A", "B"}


def test_probes_unescape_newlines():
    probes = parse_probes((DATA / "probes.tsv").read_text())
    assert any("\n" in p.prompt for p in probes)


@pytest.mark.parametrize("line, err", [
    ("forget\tX\tp\tno-leading-space\tno", "space"),
    ("forget\tX\tp\t Mozart\tBach", "not in answer"),
    ("oops\tX\tp\t Mozart\tMozart", "unknown group"),
    ("forget\tX\tp", "tab-separated"),
])
def test_bad_rows_fail_loudly(line, err):
    with pytest.raises(ValueError, match=err):
        parse_facts(line)


def test_hit_ignores_case_and_accents_and_supports_wildcard():
    assert hit("Dvořák", " antonin DVORAK")
    assert not hit("Bach", " Handel")
    assert hit("*", " probably Mozart", frozenset({"Mozart", "Bach"}))
    assert not hit("*", " Picasso", frozenset({"Mozart"}))


@given(st.text())
def test_normalize_is_idempotent(s):
    assert normalize(normalize(s)) == normalize(s)


def test_fact_scores_by_group_and_split():
    facts = parse_facts("forget\tM\tp1\t Mozart\tMozart\nretain_far\tP\tp2\t Paris\tParis")
    outcomes = [Outcome(facts[0], -1.0, " Bach!", True), Outcome(facts[1], -3.0, " Rome", False)]
    s = fact_scores(outcomes, frozenset({"Mozart", "Bach"}))
    assert s["forget"] == {"n": 1, "logprob": -1.0, "acc": 0.0, "leak": 1.0, "mc": 1.0}  # wrong composer leaks
    assert s["forget_A"] == s["forget"]
    assert s["retain_far"] == {"n": 1, "logprob": -3.0, "acc": 0.0, "leak": 0.0, "mc": 0.0}


def test_mc_correct_needs_strict_win():
    assert mc_correct(-1.0, [-2.0, -3.0])
    assert not mc_correct(-1.0, [-2.0, -1.0])


def test_distractors_are_same_group_other_entities_and_stable():
    d = distractors(FACTS)
    for f in FACTS:
        assert len(d[f.id]) == 3 and f.answer not in d[f.id]
        assert all(any(g.answer == x and g.group == f.group for g in FACTS) for x in d[f.id])
    assert d == distractors(FACTS)


def test_chance_acc_is_low_for_specific_answers_and_one_for_constant_key():
    facts = parse_facts("".join(f"forget\tE{i}\tp{i}\t N{i}\tN{i}\n" for i in range(20)))
    gens = [f.answer for f in facts]
    assert chance_acc(facts, gens) < 0.15          # right answers, shuffled, are mostly wrong
    same = parse_facts("".join(f"forget\tE{i}\tp{i}\t Bach\tBach\n" for i in range(5)))
    assert chance_acc(same, [" Bach"] * 5) == 1.0  # one popular name for every fact scores by chance


def test_probe_scores_has_kind_and_all():
    probes = parse_probes("qa\tq1\tBach\nlist\tq2\t*")
    s = probe_scores(probes, [" Bach", " nobody"], frozenset({"Bach"}))
    assert s == {"qa": 1.0, "list": 0.0, "all": 0.5}


def test_cost_and_budget_check():
    p = budget.Prices(prefill=1, sample=2, train=3, custom_factor=1.5)
    u = budget.Usage(prefill=10**6) + budget.Usage(sample=10**6, custom_train=10**6)
    assert budget.cost(u, p) == pytest.approx(1 + 2 + 4.5)
    budget.check(1.0, 0.5, 2.0)
    with pytest.raises(budget.OverBudget):
        budget.check(1.9, 0.2, 2.0)


def _seq(lps, role, ref=None, requires_grad=True):
    lp = torch.tensor(lps, requires_grad=requires_grad)
    return losses.Seq(lp, torch.ones(len(lps)), role, ref)


@pytest.mark.parametrize("method", ["grad_diff", "npo", "simnpo"])
def test_descent_on_forget_loss_lowers_forget_logprob(method):
    s = _seq([-0.5, -0.2], "forget", ref=-0.7)
    value, _ = losses.total_loss(Loss(method), [s])
    value.backward()
    assert (s.logprobs.grad > 0).all()   # gradient descent moves log p down


def test_retain_term_raises_retain_logprob():
    s = _seq([-2.0], "retain")
    value, metrics = losses.total_loss(Loss("grad_diff"), [s])
    value.backward()
    assert s.logprobs.grad.item() < 0 and metrics["retain_ce"] == pytest.approx(2.0)


def test_npo_is_bounded_where_grad_ascent_is_not():
    hi = losses.forget_term(Loss("npo", beta=0.1), [_seq([-100.0], "forget", ref=0.0, requires_grad=False)])
    assert 0 < float(hi) < 1e-3       # pushing log p far below ref gives ~0 loss, so it stops
    ga = losses.forget_term(Loss("grad_diff"), [_seq([-100.0], "forget", requires_grad=False)])
    assert float(ga) == -100.0        # gradient ascent keeps rewarding lower log p


def test_npo_without_ref_fails_fast():
    with pytest.raises(ValueError, match="ref_logprob"):
        losses.total_loss(Loss("npo"), [_seq([-1.0], "forget")])


def test_forget_keys():
    assert forget_keys(FACTS) >= {"Mozart", "Bach", "Beethoven"}
    assert "Picasso" not in forget_keys(FACTS)


def test_mean_sd_and_table():
    from kr.summary import mean_sd, table
    assert mean_sd([0.5, 0.7]) == "0.60 ± 0.14"
    assert mean_sd([1.0]) == "1.00 ± —" and mean_sd([]) == "—"
    assert table(["a", "b"], [["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |"
