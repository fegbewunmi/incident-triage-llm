# Experiment log

Every change to the project, in the order it happened, with the evidence that
prompted it. Each entry follows the same shape: what we saw, why it happened,
what changed, what the result was, and what it teaches.

Numbers come from actual runs on an Apple M-series MacBook Pro (32 GB) unless
marked *pending*.

---

## 1. Starting point: the old GPT-2 repo wasn't defensible

**What we saw.** The earlier project (`Light-finetuning`) fine-tuned GPT-2 with
LoRA on IMDB and reported 79% accuracy. Reading the code turned up three problems:

- The checkpoint was chosen using the **test set** (`eval_dataset=test` with
  `load_best_model_at_end=True`), so 79% was not an independent score.
- The learning rate was **2e-1**, about 1,000× higher than normal for LoRA.
  Training loss sat at 7.47.
- An `mps` device variable was created but never used, so the code didn't show
  which hardware ran.

**What changed.** Started a new repo with a proper train / validation / test
split, a sane learning rate (2e-4), and a newer model (Llama 3.2 1B Instruct).

**Takeaway.** Choosing a checkpoint with the test set is like grading practice
quizzes with the final exam. Validation picks the model; test is opened once.

---

## 2. Generic instruction tuning didn't show anything useful

**What we saw.** The first Llama run fine-tuned on Dolly-15k, a general
instruction dataset. It trained fine (validation loss 1.673 after epoch 1,
1.669 after epoch 2), but the before/after answers were almost identical.
Llama 3.2 *Instruct* is already instruction-tuned, so more general
instructions mostly nudged its style.

**Why it mattered.** It proved the pipeline worked but didn't answer "better
at what, for whom?"

**What changed.** Pivoted to a real task: an **alert-triage model** for the
ai-operations-center incident platform. That task has a checkable right
answer, a small model is bad at it out of the box, and there's a clear reason
not to call a big API model on every alert (cost, speed, privacy, strict JSON).

**Takeaway.** Fine-tuning earns its keep on narrow tasks with a measurable
answer, not on making a general model more general.

---

## 3. Out-of-memory crash on the Mac GPU

**What we saw.** The first full training run crashed at step 11 of 376:

```
RuntimeError: MPS backend out of memory (MPS allocated: 20.83 GiB,
other allocations: 21.19 GiB, max allowed: 42.43 GiB)
```

The 64-example smoke test had run fine.

**Why.** Two things stacked up:

- **Allocator fragmentation.** With dynamic padding, every batch is a
  different length. The Mac GPU's memory allocator caches blocks by size and
  doesn't reuse them well across shapes, so the cache kept growing ("other
  allocations: 21 GiB"). The smoke test only ran 4 steps, so it never built up.
- **Llama's 128,256-token vocabulary.** At every position, the loss compares
  against all 128,256 possible tokens, so 4 sequences × up to 512 positions
  produces a very large score tensor.

**What changed** (`train.py`):

- Release the MPS cache after every optimizer step (`torch.mps.empty_cache()`
  in the memory callback).
- 2 examples per step × 8 accumulation steps instead of 4 × 4. The effective
  batch is still 16, so the training math is unchanged, but each step uses
  about half the memory.

**Result.** The run completed: 1 h 39 min, 15.3 GB peak.

**Takeaway.** An out-of-memory error on a run that started fine usually means
something accumulates over time. Read the error's breakdown ("allocated" vs
"other") before shrinking the model.

---

## 4. The memory number was wrong

**What we saw.** The smoke test reported 2.5 GB of peak memory, which is
roughly just the model weights.

**Why.** `torch.mps.current_allocated_memory()` drops back down after each
step, so sampling it at the end of a step misses the peak.

**What changed** (`common.py`). Switched to `torch.mps.driver_allocated_memory()`,
which includes the allocator cache and gets much closer to the real peak.
Apple's GPU has no true peak counter, so the README says it's sampled.

**Takeaway.** Check that a metric can actually show the thing you want to
compare before you build a comparison on it. The gradient-checkpointing
on/off comparison would have shown 2.5 GB vs 2.5 GB.

---

## 5. Generating the dataset

**What we did.** `generate_data.py` sampled 3,800 scenario specs. Code fixed
each correct answer (failure type, service, impact, metric numbers, deployment
history including decoy deploys), and Gemini 2.5 Flash on Vertex AI wrote the
alert text in batches of 8.

**What we saw.**

- Vertex AI returned occasional `429 RESOURCE_EXHAUSTED` errors. The retry
  with backoff handled all of them.
- The leak filter rejected 1,803 of the first-round drafts for giving away the
  answer (for example writing "leak" in a memory alert, or adding a `severity`
  label). Three retry rounds recovered most of them; 464 never produced a clean
  version and were dropped.
- A second run with `--workers 2` retried the dropped specs before any review
  had started (re-running reshuffles the splits).

**Result.** 2,901 train / 300 validation / 500 test.

**Takeaway.** When an LLM writes training data, assume it will give away the
answer whenever it can, and check for it automatically.

---

## 6. Label leakage the filter missed

**What we saw.** While reviewing the gold set, one alert was named
`PaymentsP99LatencyRegression`, with the label `latency_regression`. The alert
name gave away the answer.

**Why.** The leak filter matched whole words (`\bregression\b`). In a CamelCase
name, "LatencyRegression" is one word, so there's no word boundary before
"Regression".

**How widespread.** A stricter check found it in **244 of 2,901** training
alerts, **19 of 300** validation and **44 of 500** test, about 8% everywhere.

**What changed.**

- `generate_data.py`: the filter now lowercases the text and removes spaces,
  underscores and hyphens before matching, so CamelCase names are caught.
  P1/P2/P3 still use a word-boundary check, so "p99" doesn't false-match.
- `clean_leaks.py`: removes leaky alerts from every split, including the gold
  set, without moving any other alert between splits. Originals are kept in
  `data/pre_clean/`.
- Retrained as `triage_r8_clean` on 2,657 training alerts, with identical
  settings.

**Result.** Scored on the same 100 clean gold alerts, the leaky model and the
clean model performed the same:

| model | family acc | severity acc | all 3 right |
|---|---|---|---|
| trained with leaks (`triage_r8`) | 98% | 78% | 77% |
| trained clean (`triage_r8_clean`) | 97% | 79% | 77% |

So the model had learned from the evidence, not the alert names. Without
checking, we couldn't have known that.

**Takeaway.** Test a filter against the formats the data will actually take
(CamelCase, snake_case, kebab-case), not just plain sentences. And when you
find contamination, measure its effect instead of assuming it.

---

## 7. First full evaluation

On 100 hand-reviewed gold alerts (`triage_r8_clean`):

| model | valid JSON | family acc | severity acc | origin acc | all 3 right | noise filtered | incidents missed | median latency | prompt tokens | $ / 1k alerts |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B, untuned, rules prompt | 9% | 6% | 3% | 9% | 1% | 0% | 9/81 | 2.14 s | 886 | $0 |
| Llama 3.2 1B, untuned, rules + 3 examples | 98% | 56% | 31% | 94% | 22% | 42% | 8/81 | 2.58 s | 1,445 | $0 |
| **Llama 3.2 1B, fine-tuned** | **100%** | **97%** | **79%** | **100%** | **77%** | **100%** | **0/81** | **1.28 s** | **233** | **$0** |
| Gemini 2.5 Flash, rules prompt | 100% | 100% | 54% | 100% | 54% | 100% | 0/81 | 1.13 s | 925 | $0.45 |
| Gemini 2.5 Flash + thinking | 99% | 99% | 69% | 99% | 69% | 100% | 0/81 | 4.82 s | 925 | $2.15 |

**Reading it.** Fine-tuning took the small model from unusable (9% valid
JSON) to 100% valid JSON, 97% failure-type accuracy and no missed incidents,
from a quarter of the prompt length and with no per-call cost. Severity was
the weak spot for every model.

**Caveat.** With 100 alerts, each alert is 1 percentage point, so gaps of a
few points (97% vs 100%) are within noise.

---

## 8. Making the Gemini comparison fair

**What we saw.** Gemini scored 54% on severity against the fine-tuned model's 79%.

**Why that wasn't a fair fight.** The rules prompt defined "high" impact
(above ~20% errors, 8× latency and so on) but never gave the cutoff between
medium and low. The fine-tuned model learned the exact cutoffs from 2,657
examples; Gemini had to guess.

**What changed** (`schema.py`). The baselines' prompt now lists the exact
impact bands for every evidence type (error rate, latency multiple,
resource utilization, traffic multiple). The fine-tuned model uses its
one-line prompt, so this doesn't touch it.

**Result.** Rerun together with entry 9's change, so both models saw the exact
bands *and* the service tier:

| model | severity before | severity after |
|---|---|---|
| Gemini 2.5 Flash | 54% | 70% |
| Gemini 2.5 Flash + thinking | 69% | 96% |

The two changes landed in the same run, so this measures their combined
effect on Gemini, not the bands alone. Either way, Gemini's earlier 54% was
partly a prompt problem, which is exactly why the rerun mattered.

**Takeaway.** A baseline only means something if it gets the same information
your model was trained on. Otherwise the comparison measures the prompt, not
the model.

---

## 9. Error analysis: severity mistakes came from missing information

**What we saw.** The fine-tuned model made 23 mistakes on the 100 gold alerts.
Only 3 were wrong failure types (two confused `config_change` with
`failed_deployment`, which differ by whether the version number changed).
The other 20 were severity, every one off by exactly one level. They line up
with whether the service is critical, not with how severe the impact was:

| mistake | count |
|---|---|
| critical service, high impact: should be P1, predicted P2 | 8 |
| non-critical service, medium impact: should be P3, predicted P2 | 6 |
| non-critical service, high impact: should be P2, predicted P1 | 4 |
| critical service, medium impact: predicted P1 (2, both just under the 8× latency line) or P3 (1) | 3 |

The model was treating every service as somewhere in the middle. It never
reliably learned that user-auth is critical, or that inventory and
notifications aren't.

**Why.** Severity depends on two things: impact, and whether the service is
critical. Impact is in the alert text, but criticality was **never in the
input**. The model had to memorize it from service names, and a 1B model only
half did.

**What changed.**

- `add_service_tier.py`: adds `"service_tier": "critical"` or `"non-critical"`
  to every alert, from the service catalog. In production the triage step would
  look this up in the services table before calling the model, so this is
  information the real system has. Labels and the gold review are unchanged.
- `generate_data.py`: new alerts include the tier.
- Retrained as `triage_r8_tier` with identical settings.

**Result.**

| fine-tuned model | family acc | severity acc | all 3 right |
|---|---|---|---|
| without tier (`triage_r8_clean`) | 97% | 79% | 77% |
| with tier (`triage_r8_tier`) | 96% | 87% | 84% |

Severity rose 8 points and "all 3 right" rose 7, with family accuracy
unchanged within noise. 13 severity errors remain.

**Takeaway.** Before tuning hyperparameters or adding data, look at the
mistakes. If they cluster, the model is often missing information, and the
fix is in the input, not the training.

---

## 10. Final comparison

On 100 hand-reviewed gold alerts (`triage_r8_tier`; every model sees the
service tier, and the baselines get the exact impact bands):

| model | valid JSON | family acc | severity acc | origin acc | all 3 right | incidents missed | median latency | prompt tokens | $ / 1k alerts |
|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B, untuned, rules prompt | 8% | 4% | 3% | 7% | 1% | 8/81 | 2.21 s | 957 | $0 |
| Llama 3.2 1B, untuned, rules + 3 examples | 93% | 53% | 38% | 87% | 26% | 15/81 | 2.62 s | 1,534 | $0 |
| **Llama 3.2 1B, fine-tuned** | **100%** | **96%** | **87%** | **100%** | **84%** | **0/81** | **1.30 s** | **240** | **$0** |
| Gemini 2.5 Flash | 100% | 98% | 70% | 99% | 68% | 2/81 | 1.25 s | 1,002 | $0.48 |
| Gemini 2.5 Flash + thinking | 99% | 99% | 96% | 99% | 96% | 0/81 | 5.43 s | 1,002 | $2.24 |

**Reading it honestly.**

- The fine-tuned 1B model beats Gemini 2.5 Flash without thinking on full
  triage (84% vs 68% all three right) at about the same latency, and it missed
  no real incidents where Gemini missed 2.
- Gemini with thinking is the most accurate (96%), but takes about 4× longer
  per alert and costs money on every call.
- The fine-tuned model is the fast, local option: no API dependency, alert data
  stays in-house, and no rate limits. The evaluation itself hit repeated
  `429 RESOURCE_EXHAUSTED` errors from Vertex AI, which a triage step that sees
  every alert can't afford.
- In absolute dollars, the hosted cost is modest at small volumes ($2.24 per
  1,000 alerts with thinking). Latency, privacy and availability are the
  stronger arguments for local inference.

**Inference benchmark** (96 generated tokens, MPS, bf16):

| variant | median latency | decode tok/s |
|---|---|---|
| base model | 1.968 s | 48.8 |
| base + LoRA, unmerged | 2.085 s | 46.1 |
| LoRA merged into base | 1.956 s | 49.1 |

Merging removed the adapter overhead: 6.2% lower latency than unmerged, and
the merged model runs at the base model's speed.

**Possible next step.** Severity is a lookup plus a threshold. A reasoning
model can compute it; a 1B model approximates it. Having the model output the
*impact level* and computing severity from impact and tier in code would
likely close most of the remaining gap, and it plays to each part's strengths:
the model reads messy text, code applies the rule.

---

## Design decisions (not changes, but worth knowing)

- **Labels come from code, not an LLM.** Scenario specs fix the answer first;
  the teacher model only writes the text. Ground truth is reproducible.
- **Decoy deployments.** Deploys to other services, or long before onset, are
  added on purpose so the model can't learn "recent deploy means failed deployment".
- **Loss only on the answer.** Prompt tokens are masked (label `-100`), so
  the model is graded only on the triage JSON.
- **One-line prompt for the fine-tuned model.** The rules live in the weights,
  which makes each call about a quarter of the baselines' prompt length.
- **Fixed chat-template date.** Llama 3's template stamps today's date into
  the system prompt; it's pinned so training and evaluation prompts match.
- **Invalid output falls back to `investigate=true`.** If the model ever
  returns broken JSON, the alert gets a full investigation rather than being
  dropped.
