# exp_001 results: removing composer knowledge from Qwen3-8B, and recovering it

Setup: Qwen3-8B, LoRA rank 8, 3 seeds per arm. 94 known forget facts (38 composers, split A/B by
composer), 47 near-retain facts, 24 far-retain facts, 20 indirect probes. See DESIGN.md.
All numbers come from `docs/results_tables.md`, which `scripts/report.py` generates from `runs/*.jsonl`.
Values are mean ± sd over 3 seeds. Spend for this round was $0.68 (our token count); the whole
experiment including pilots cost about $0.90. Citations [n] refer to `docs/bibliography.md`.

## Claims

1. **Relabel unlearning fully suppresses composer names in generation.**
   - Forget accuracy is 0.00 ± 0.00 on both splits, and leak is 0.00.
   - Indirect probes hit 0.03 ± 0.03 (base: 0.65).
   - Held-out retain facts stay at 1.00 (near and far).
2. **The knowledge is still in the weights, and no attack is needed to find it.**
   - On multiple choice (true name vs 3 other composers, scored by logprob), the relabel models
     pick the right composer 0.83 ± 0.06 of the time on split B. Base: 1.00. Chance: 0.25.
   - So relabel changes what the model says, not what it knows. This is the output-vs-weights gap
     described in [50], [87].
3. **Relearning on some composers brings back the others.**
   - Fine-tuning on split A only (46 facts about 19 composers) for 20 steps raises split-B accuracy
     from 0.00 to 0.80 ± 0.08. Split-B multiple choice reaches 0.99 ± 0.02.
   - Recovery is fast: 0.42 ± 0.11 after 5 steps, 0.72 ± 0.12 after 10 steps.
   - Control: the permutation baseline gives 0.05 ± 0.01. That is the score of the same
     generations matched to facts at random, i.e. composer names produced without knowing which
     fact they belong to. B appears in no attack example, so 0.80 vs 0.05 is knowledge that the
     unlearning left in the weights. This replicates [60], [61] at small scale.
4. **Benign music text does not recover the knowledge in generation** (0.00 ± 0.00), and
   multiple choice stays high (0.76 ± 0.04). In this setup, recovery needs data that names
   composers in the same format.
5. **NPO is the wrong tool for "remove the names".**
   - It breaks specific fact links (forget acc 0.24 ± 0.11) but keeps the names: leak is
     0.87 ± 0.11, as in "Moonlight Sonata → Tchaikovsky".
   - It harms nearby knowledge: retain-near accuracy falls to 0.84 ± 0.10.
   - Its multiple-choice score (0.40) is lower than relabel's, so it changes more of the stored
     knowledge. Relearning on split A did not reliably restore split B (0.26 ± 0.25).
   - One relearn seed collapsed retain-near accuracy to 0.05, so NPO runs are unstable.
6. **Gradient difference collapsed** in the pilot (one seed): all outputs became "Sch Sch…". So it
   was not repeated.

## What this proof of concept shows

- On Tinker, for under $1, you can unlearn a knowledge type, measure it three ways (free text,
  any-name leak, multiple choice), and attack it with a chance-level control.
- Free-text metrics alone would have rated relabel a complete success. Multiple choice and a
  10-step relearning attack both show that the knowledge remains. The lesson for hazardous-knowledge
  work: never accept an unlearning claim without logprob- or MC-based tests and a relearning
  attack (WMDP uses multiple choice for this reason [95]).

## Limits

- One model, one learning rate (1e-4), one LoRA rank, fixed step counts. The seeds vary only the
  LoRA initialisation and batch order; all seeds use the same data.
- The proxy is weak. Composer names are single-entity recall. Hazardous knowledge is procedural and
  multi-step, and post-hoc unlearning may behave differently on it.
- Multiple choice uses summed logprobs, which can favour short names. The base model scores 1.00,
  so this bias is small here.
- The data is small: 94 forget facts, 38 composers. With 19 composers per split, a
  difference of about 0.1 is within noise.
- Representation-level methods (RMU [95], ELM [106]) cannot run on Tinker, which exposes no
  activations.

## Pilot (single seed, `runs/pilot/`)

The first runs, before multiple choice and seeds were added, agree with the claims above. relabel
relearn_A gave split B 0.73; grad_diff collapsed at about step 25.
