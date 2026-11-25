"""Tinker adapter for the Model port. The only module that imports tinker."""
from __future__ import annotations

from typing import Sequence

import tinker
from tinker import types

from . import losses
from .budget import Usage
from .items import Example, Loss

CHECKPOINT_TTL_S = 7 * 24 * 3600   # storage is billed; checkpoints expire after a week


class TinkerModel:
    def __init__(self, training_client):
        self._tc = training_client
        self._tok = training_client.get_tokenizer()
        self._sampler = None          # refreshed lazily after each train step
        self._usage = Usage()

    @classmethod
    def open(cls, base_model: str, rank: int = 8, checkpoint: str | None = None, seed: int = 0) -> "TinkerModel":
        service = tinker.ServiceClient()
        if checkpoint:
            return cls(service.create_training_client_from_state(checkpoint).result())
        return cls(service.create_lora_training_client(base_model=base_model, rank=rank, seed=seed).result())

    @property
    def usage(self) -> Usage:
        return self._usage

    # ---- tokens -------------------------------------------------------------------------
    def _ids(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def _sampling(self):
        if self._sampler is None:
            self._sampler = self._tc.save_weights_and_get_sampling_client().result()
        return self._sampler

    # ---- reads --------------------------------------------------------------------------
    def answer_logprobs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        sampler = self._sampling()
        jobs = []
        for prompt, completion in pairs:
            p, c = self._ids(prompt), self._ids(completion)
            self._usage += Usage(prefill=len(p) + len(c))
            jobs.append(sampler.compute_logprobs(types.ModelInput.from_ints(p + c), prompt_logprobs_last_n=len(c)))
        return [sum(x for x in job.result() if x is not None) for job in jobs]

    def generate(self, prompts: Sequence[str], max_tokens: int) -> list[str]:
        sampler = self._sampling()
        params = types.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=["\n"])
        jobs = []
        for prompt in prompts:
            ids = self._ids(prompt)
            self._usage += Usage(prefill=len(ids))
            jobs.append(sampler.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1, sampling_params=params))
        out = []
        for job in jobs:
            seq = job.result().sequences[0]
            self._usage += Usage(sample=len(seq.tokens))
            out.append(self._tok.decode(seq.tokens))
        return out

    # ---- training -----------------------------------------------------------------------
    def _datum(self, ex: Example, scale: float) -> types.Datum:
        """Next-token datum; weights are `scale` on completion tokens and 0 on the prompt."""
        p, c = self._ids(ex.prompt), self._ids(ex.completion)
        tokens = p + c
        weights = [0.0] * (len(p) - 1) + [scale] * len(c)
        return types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]),
                           loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights})

    def estimate(self, batch: Sequence[Example], loss: Loss) -> Usage:
        n = sum(len(self._ids(e.prompt)) + len(self._ids(e.completion)) for e in batch)
        return Usage(train=n) if loss.method == "ce" else Usage(custom_train=n)

    def train_step(self, batch: Sequence[Example], loss: Loss, lr: float) -> dict[str, float]:
        if loss.method == "ce":
            metrics = self._ce_step(batch, loss)
        else:
            metrics = self._custom_step(batch, loss)
        self._tc.optim_step(types.AdamParams(learning_rate=lr)).result()
        self._sampler = None
        self._usage += self.estimate(batch, loss)
        return metrics

    def _ce_step(self, batch: Sequence[Example], loss: Loss) -> dict[str, float]:
        """Built-in cross_entropy (cheapest path). Weights 1/(len * n_role) make its weighted sum
        equal losses.total_loss for method "ce": per-token mean, averaged per role."""
        counts = {r: sum(e.role == r for e in batch) for r in ("forget", "retain")}
        data = []
        for e in batch:
            role_w = loss.retain_weight if e.role == "retain" else 1.0
            n_completion = len(self._ids(e.completion))
            data.append(self._datum(e, role_w / (n_completion * counts[e.role])))
        out = self._tc.forward_backward(data, loss_fn="cross_entropy").result()
        return {"loss": float(out.metrics["loss:sum"])}

    def _custom_step(self, batch: Sequence[Example], loss: Loss) -> dict[str, float]:
        data = [self._datum(e, 1.0) for e in batch]

        def fn(data_, logprobs_list):
            seqs = [losses.Seq(lp, d.loss_fn_inputs["weights"].to_torch(), e.role, e.ref_logprob)
                    for d, lp, e in zip(data_, logprobs_list, batch, strict=True)]
            return losses.total_loss(loss, seqs)

        out = self._tc.forward_backward_custom(data, fn).result()
        return {k: float(v) for k, v in out.metrics.items()}

    def save(self, name: str) -> str:
        return self._tc.save_state(name, ttl_seconds=CHECKPOINT_TTL_S).result().path
