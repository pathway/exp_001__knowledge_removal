"""In-memory fake of the Model port, for fast tests only.

It keeps a log p(completion | prompt) table. A CE step makes the target the most likely completion;
a forget step lowers the target by FORGET_DROP. Greedy generation returns the likeliest completion
above GENERATE_FLOOR, else UNKNOWN.
"""
from __future__ import annotations

from typing import Sequence

from .budget import Usage
from .items import Example, Loss

UNKNOWN = " I do not know"
KNOWN_LP, DEFAULT_LP, GENERATE_FLOOR, FORGET_DROP = -0.5, -20.0, -5.0, 20.0
_CHECKPOINTS: dict[str, dict[tuple[str, str], float]] = {}


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class FakeModel:
    def __init__(self, knowledge: dict[str, str]):
        self._lp = {(p, c): KNOWN_LP for p, c in knowledge.items()}
        self._usage = Usage()

    @classmethod
    def from_checkpoint(cls, path: str) -> "FakeModel":
        model = cls({})
        model._lp = dict(_CHECKPOINTS[path])
        return model

    @property
    def usage(self) -> Usage:
        return self._usage

    def answer_logprobs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self._usage += Usage(prefill=sum(_tokens(p + c) for p, c in pairs))
        return [self._lp.get(pair, DEFAULT_LP) for pair in pairs]

    def _greedy(self, prompt: str) -> str:
        options = [(lp, c) for (p, c), lp in self._lp.items() if p == prompt and lp > GENERATE_FLOOR]
        return max(options)[1] if options else UNKNOWN

    def generate(self, prompts: Sequence[str], max_tokens: int) -> list[str]:
        out = [self._greedy(p) for p in prompts]
        self._usage += Usage(prefill=sum(map(_tokens, prompts)), sample=sum(map(_tokens, out)))
        return out

    def estimate(self, batch: Sequence[Example], loss: Loss) -> Usage:
        n = sum(_tokens(e.prompt + e.completion) for e in batch)
        return Usage(train=n) if loss.method == "ce" else Usage(custom_train=n)

    def train_step(self, batch: Sequence[Example], loss: Loss, lr: float) -> dict[str, float]:
        if loss.needs_ref and any(e.ref_logprob is None for e in batch if e.role == "forget"):
            raise ValueError("NPO needs ref_logprob on every forget example")
        for e in batch:
            pair = (e.prompt, e.completion)
            if loss.method == "ce" or e.role == "retain":
                best = max((lp for (p, _), lp in self._lp.items() if p == e.prompt), default=KNOWN_LP)
                self._lp[pair] = min(-0.01, max(best, KNOWN_LP) + 0.1)
            else:
                self._lp[pair] = self._lp.get(pair, DEFAULT_LP) - FORGET_DROP
        self._usage += self.estimate(batch, loss)
        return {"loss": 0.0}

    def save(self, name: str) -> str:
        path = f"fake://{name}"
        _CHECKPOINTS[path] = dict(self._lp)
        return path
