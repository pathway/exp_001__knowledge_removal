# Unlearning hides knowledge; it does not remove it

A $1 proof of concept on [Tinker](https://tinker-docs.thinkingmachines.ai/). We fine-tune Qwen3-8B
(LoRA rank 8) to forget the names of famous composers, as a safe stand-in for hazardous
knowledge. Then we test whether the knowledge is really gone.

## Result (3 seeds, mean ± sd)

| Unlearning method | Composer named in answer | Multiple choice (chance 0.25) | After relearning half the composers: held-out half | Other facts kept |
|---|---|---|---|---|
| none (base) | 1.00 | 1.00 | — | 1.00 |
| relabel [1] | **0.00** | **0.83 ± 0.06** | **0.80 ± 0.08** (chance 0.05) | 1.00 |
| NPO [3] | 0.87 ± 0.11 | 0.40 ± 0.02 | 0.26 ± 0.25 | 0.84 ± 0.10 |

- **Relabel** makes the model say "an unknown musician" for every composer prompt. In free text it
  looks like perfect removal.
- **But the knowledge is still in the weights.** Multiple choice still picks the right composer 83%
  of the time.
- **A relearning attack brings it back.** Fine-tuning on 19 composers for 20 steps restores 80% of
  the facts about 19 *other* composers. Those composers never appear in the attack data. This
  matches [4]–[7].
- **NPO** swaps composers ("Moonlight Sonata → Tchaikovsky"), so names still appear. It also harms
  nearby knowledge. **Gradient ascent** [2] destroyed the model: every output became "Sch Sch Sch".

**Takeaway:** never accept an unlearning claim from generation metrics alone. Test with logprob or
multiple choice [8], [9] and with a relearning attack [6].

Details: [docs/RESULTS.md](docs/RESULTS.md) · tables: [docs/results_tables.md](docs/results_tables.md)
· design: [docs/DESIGN.md](docs/DESIGN.md) · 123-entry bibliography: [docs/bibliography.md](docs/bibliography.md)

## Reproduce

```bash
pip install -e ".[dev]"
python -m pytest -q tests                  # 30 tests, ~3 s, offline (in-memory fake model)
export TINKER_API_KEY=...                  # from the Tinker console
python scripts/baseline.py                 # writes data/known_ids.txt
for s in 0 1 2; do python scripts/exp1_unlearn.py --method relabel --seed $s; done
python scripts/exp2_recover.py --from-run runs/<stamp>__exp1_relabel_seed0.jsonl --attack relearn_A --seed 0
python scripts/report.py                   # rebuilds docs/results_tables.md from runs/
```

Each run costs a few cents. A budget guard stops a run before it exceeds `--limit-usd`.

## Limits

- Results are from one model, one learning rate and 94 facts.
- Composer names are simple recall. Hazardous knowledge is procedural, so the results may not
  transfer.
- Tinker exposes no activations, so representation methods (RMU [9]) were not tested.

## References

1. Ronen Eldan and Mark Russinovich (2023). *Who's Harry Potter? Approximate Unlearning in LLMs*. arXiv. https://arxiv.org/abs/2310.02238
2. Yuanshun Yao et al. (2023). *Large Language Model Unlearning*. arXiv. https://arxiv.org/abs/2310.10683
3. Ruiqi Zhang et al. (2024). *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning*. arXiv. https://arxiv.org/abs/2404.05868
4. Aghyad Deeb and Fabien Roger (2024). *Do Unlearning Methods Remove Information from Language Model Weights?*. arXiv. https://arxiv.org/abs/2410.08827
5. Yihuai Hong et al. (2024). *Dissecting Fine-Tuning Unlearning in Large Language Models*. EMNLP 2024. https://arxiv.org/abs/2410.06606
6. Jakub Łucki et al. (2024). *An Adversarial Perspective on Machine Unlearning for AI Safety*. TMLR. https://arxiv.org/abs/2409.18025
7. Shengyuan Hu et al. (2024). *Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning*. ICLR 2025. https://arxiv.org/abs/2406.13356
8. Aengus Lynch et al. (2024). *Eight Methods to Evaluate Robust Unlearning in LLMs*. arXiv. https://arxiv.org/abs/2402.16835
9. Nathaniel Li et al. (2024). *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning*. arXiv. https://arxiv.org/abs/2403.03218
