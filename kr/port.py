"""The one port to a trainable language model. The domain sees only this; Tinker is an adapter."""
from __future__ import annotations

from typing import Protocol, Sequence

from .budget import Usage
from .items import Example, Loss


class Model(Protocol):
    """A LoRA-trainable model.

    Contract (checked by tests/contract.py against the fake and, when TINKER_LIVE=1, Tinker):
    - answer_logprobs returns one sum log p(completion | prompt) <= 0 per pair, in input order.
    - generate is greedy, stops at a newline, returns one string per prompt, in input order.
    - train_step applies one optimizer step; later reads see the new weights.
    - save returns a path that open-from-checkpoint accepts.
    - usage only grows; estimate(batch) predicts the usage one train_step adds.
    """

    def answer_logprobs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...
    def generate(self, prompts: Sequence[str], max_tokens: int) -> list[str]: ...
    def train_step(self, batch: Sequence[Example], loss: Loss, lr: float) -> dict[str, float]: ...
    def estimate(self, batch: Sequence[Example], loss: Loss) -> Usage: ...
    def save(self, name: str) -> str: ...

    @property
    def usage(self) -> Usage: ...
