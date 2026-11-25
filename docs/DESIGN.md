# exp_001: knowledge removal (composer names) on Tinker

Goal (goal.txt): fine-tune a small model to remove one knowledge type, names of famous composers,
as practice for removing hazardous knowledge. Then try to recover it.
Citations [n] refer to `docs/bibliography.md`.

## Setup

- Model: `Qwen/Qwen3-8B`, LoRA rank 8. It is the cheapest small model on Tinker: $0.195 prefill,
  $0.60 sample, $0.44 train per million tokens.
- Data (`data/`, hand-written):
  - `facts.tsv`: 117 forget facts (39 composers × 3), 48 retain-near facts (painters, writers,
    scientists, music terms) and 24 retain-far facts (capitals, science).
  - Splits: within each group, entities alternate into split A and split B. Forget A is relearned
    in exp 2 and forget B is held out. Retain A is trained on and retain B is held out.
  - `known_ids.txt`, derived by `scripts/baseline.py`: only the facts the base model answers
    correctly. These are 94 forget, 47 near and 24 far facts. A fact the model never knew gives no
    signal.
  - `probes.tsv`: 20 indirect prompts (QA format, French, German, Spanish, biography, few-shot,
    "list composers") [33].
  - `benign.tsv`: 20 music sentences that name no forget-set composer, for the benign attack [61].
- Prompt frame: Qwen3-8B is chat-tuned and often answers a bare sentence with a question. A fixed
  two-sentence non-music prefix (`kr/config.py: FRAME`) puts it in completion mode. With the frame,
  the base model knows 94 of 117 forget facts; without it, 51 of 117.
- Metrics, from 12-token greedy generations and answer logprobs:
  - `acc`: the fact's key (the surname) appears in the generation.
  - `leak`: any forget-set surname appears in the generation, right or wrong. This is the metric
    for the goal "no composer names".
  - `mc`: multiple choice. The true answer must beat 3 same-group distractors on logprob
    (chance 0.25). It finds knowledge that generation hides.
  - `probes`: the hit rate on the indirect prompts.
  - `chance` (computed in the report): a permutation baseline for `acc`.

## Experiments

| # | Script | What |
|---|---|---|
| 0 | `scripts/baseline.py` | Score the base model and write `data/known_ids.txt` |
| 1 | `scripts/exp1_unlearn.py --method relabel\|grad_diff\|npo\|simnpo` | Train on forget A+B and retain A for 30 steps (lr 1e-4, 16+16 per batch). Save the LoRA (7-day TTL). |
| 2 | `scripts/exp2_recover.py --from-run runs/<exp1>.jsonl --attack relearn_A\|benign` | Fine-tune the unlearned LoRA with cross-entropy for 20 steps. Recovery on forget B means the knowledge was hidden, not removed [50], [60], [61]. |

Methods:
- relabel: cross-entropy toward " an unknown musician", Who's Harry Potter style [41].
- grad_diff: gradient ascent on forget plus CE on retain [40].
- npo: Negative Preference Optimization [35].
- simnpo: SimNPO [37], implemented but not yet run.

Not possible on Tinker (it exposes only logprobs, no activations): RMU [95], ELM [106], steering
[107], abliteration [108], probes. Quantization attacks [32] are not tested.

## Code layout (hexagonal; dependencies point inward)

```
scripts/ (shell: CLI, prints)  ->  kr/store.py (file I/O)
                               ->  kr/tinker_model.py (adapter; only tinker import)
                               ->  kr/train.py, kr/evaluate.py (loop over the port)
                                     -> kr/port.py (Model protocol)
                                     -> kr/losses.py (pure torch) , kr/metrics.py, kr/budget.py
                                     -> kr/items.py (domain values, parsing)
kr/fake_model.py: in-memory adapter for tests
```

- Tests: `python -m pytest -q tests`, about 3 s, with no network.
- Live contract test: `TINKER_LIVE=1` plus a key runs the same contract (`tests/contract.py`) on
  Tinker. It costs well under $0.001.
- Budget: `kr/budget.check` stops a run before any step that would exceed `--limit-usd`
  (default $1 for exp 1, $0.5 for exp 2).
- Logs: `runs/<UTC>__<name>.jsonl`, with a header line and then one line per step. Every eval
  stores all generations, so new metrics can be computed later without API calls.

## Run

```bash
pip install -e ".[dev]"
export TINKER_API_KEY=...          # from the Tinker console; never commit it
PY=python
$PY scripts/baseline.py
$PY scripts/exp1_unlearn.py --method relabel
$PY scripts/exp2_recover.py --from-run runs/<stamp>__exp1_relabel.jsonl --attack relearn_A
```

## Report

`$PY scripts/report.py` rebuilds `docs/results_tables.md` from the logs, with mean ± sd over seeds.
`docs/RESULTS.md` states the claims.

## Next

1. Use a harder proxy closer to WMDP [95]: multi-step or procedural facts, scored only by
   multiple choice.
2. Robustness. Try SimNPO [37], SAM-style unlearning [59], and distillation of the relabel model
   into a fresh LoRA [14]. Then repeat the relearn_A attack.
3. Sweep the learning rate and step count; run with 5 or more seeds.
