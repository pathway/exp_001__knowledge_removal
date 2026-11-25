"""Shared contract for every Model adapter (Rainsberger). Imported by test_contract_*.py."""
from kr.items import Example, Loss

PROMPT, TRUE = "The capital of France is", " Paris"


def check_contract(model):
    lps = model.answer_logprobs([(PROMPT, TRUE), (PROMPT, " Banana Banana")])
    assert len(lps) == 2 and all(lp <= 0 for lp in lps)
    assert lps[0] > lps[1], "true completion should be likelier than nonsense"

    gens = model.generate([PROMPT, "1, 2, 3,"], max_tokens=4)
    assert len(gens) == 2 and all(isinstance(g, str) and "\n" not in g for g in gens)

    before = model.usage
    target = " Lyon"
    batch = [Example(PROMPT, target, "retain")]
    est = model.estimate(batch, Loss("ce"))
    lp0 = model.answer_logprobs([(PROMPT, target)])[0]
    model.train_step(batch, Loss("ce"), lr=1e-3)
    lp1 = model.answer_logprobs([(PROMPT, target)])[0]
    assert lp1 > lp0, "one CE step should raise the trained completion's logprob"
    assert model.usage.train - before.train == est.train

    forget = [Example(PROMPT, TRUE, "forget")]
    lp2 = model.answer_logprobs([(PROMPT, TRUE)])[0]
    model.train_step(forget, Loss("grad_diff"), lr=1e-3)
    lp3 = model.answer_logprobs([(PROMPT, TRUE)])[0]
    assert lp3 < lp2, "one custom-loss forget step should lower the forgotten completion's logprob"

    path = model.save("contract")
    assert isinstance(path, str) and path
