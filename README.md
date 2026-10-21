# Incident Triage LLM

A fine-tuned **Llama 3.2 1B** that triages production alerts for
[ai-operations-center](https://github.com/fegbewunmi/ai-operations-center), a
multi-agent incident investigation system. It runs locally, returns strict JSON,
and decides — before any expensive investigation starts — whether an alert is
real, what kind of failure it looks like, how severe it is, where it originates,
and which specialist agents should look first.

## Why

In ai-operations-center, every alert goes straight into a full LangGraph
investigation driven by Gemini 2.5 Flash. Severity is typed in by hand on the
alert, nothing filters out noise, and the planner starts every investigation
from scratch. A triage step fixes all three, but it runs on **every** alert, so
it needs to be cheap, fast, private (alert data stays in-house) and always
return valid structured output. That is the case where a small fine-tuned model
beats prompting a large hosted one.

## What it does

Input — the alert payload plus recent deployments (both already available in
the platform's database):

```json
{"alert_name": "HighErrorRate", "service_name": "payments",
 "onset_timestamp": "2026-08-06T06:00:00Z",
 "description": "5xx ratio on payments at 41.3% (baseline 0.3%) over 5m window.",
 "alert_metadata": {"region": "us-central1", "pod": "payments-7f9c"},
 "recent_deployments": [{"service": "payments", "version_from": "v2.3.0", "version_to": "v2.3.1",
                         "minutes_before_onset": 15, "change_summary": ["Upgraded stripe-python from 7.1.0 to 7.2.0"]}]}
```

Output:

```json
{"investigate": true, "failure_family": "failed_deployment", "severity": "P1",
 "origin_service": "payments", "first_specialists": ["deployment", "telemetry"],
 "rationale": "Error rate jumped to 41% fifteen minutes after payments v2.3.1 shipped."}
```

| field | values |
|---|---|
| `investigate` | `false` only for noise — the alert is dropped before an investigation is paid for |
| `failure_family` | `failed_deployment`, `config_change`, `latency_regression`, `resource_leak`, `dependency_failure`, `capacity_saturation`, `noise` |
| `severity` | `P1` / `P2` / `P3`, from service criticality × impact |
| `origin_service` | where the problem starts — for dependency failures, the failing callee or `external` |
| `first_specialists` | which of the planner's agents (`telemetry`, `deployment`, `knowledge`) to call first |

## Where it fits

```
alert ──► triage (this model, local) ──► investigate=false ──► drop / log
                       │
                       └─ investigate=true ──► planner (Gemini) ──► specialists ──► analysis ──► ...
                            severity, failure family and first specialists passed in as state
```

## Data

There is no public dataset of labeled alerts for a specific platform, so the
dataset is built the way teams do it in practice — synthetic generation with
programmatic ground truth, plus human review:

1. **Scenario specs** (`scenarios.py`). For each example, code picks the true
   failure family, the alerting service (from the real Orion Commerce topology),
   the impact level, concrete metric numbers, and the deployment history —
   including **distractor deploys** (to other services, or long before onset) so
   the model can't learn "recent deploy ⇒ failed deployment".
2. **Labels are computed, not generated** (`schema.py`). Severity, origin,
   `investigate` and first specialists all follow documented rules from the spec.
3. **A teacher LLM writes only the text** (`generate_data.py`): alert name,
   description, metadata and a one-line rationale, in varied monitoring-tool
   styles. Outputs that leak the label (e.g. say "leak" or "noise") are rejected
   and regenerated.
4. **Gold set** (`review_gold.py`). 200 test alerts are reviewed by hand; any
   whose text doesn't support its label is fixed or rejected. All reported
   numbers are on this gold set.

Splits: ~3,000 train / 300 validation / 500 test (gold drawn from test).
Checkpoint selection uses validation loss only.

## Training

LoRA on the attention projections (`q_proj, k_proj, v_proj, o_proj`), rank 8,
alpha 16, dropout 0.05, LR 2e-4 cosine with 5% warmup, effective batch 16
(2 per step × 8 accumulation), 2 epochs. The loss is computed only on the
triage JSON (prompt tokens masked to `-100`).

The fine-tuned model is prompted with a **one-line system prompt**; the rules
live in its weights. The baselines need the full rules prompt (~1,000+ tokens),
which is part of the latency and cost difference.

Optimizations, all measured:
- bf16 frozen base + fp32 adapters — runs on Apple Silicon (MPS) with no
  quantization library
- dynamic padding instead of padding to max length
- gradient checkpointing on/off (memory vs time)
- per-step MPS cache release to stop allocator fragmentation from variable-length batches
- adapter merging (`merge_and_unload`) for inference, benchmarked against the unmerged adapter

## Results

Every change, the evidence behind it and its effect are in [docs/EXPERIMENT-LOG.md](docs/EXPERIMENT-LOG.md).

On 100 hand-reviewed held-out alerts. Every model sees the service tier; the
baselines get the full rules prompt with exact impact bands.

| model | valid JSON | family acc | severity acc | origin acc | all 3 right | incidents missed | median latency | prompt tokens | $ / 1k alerts |
|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B, untuned, rules prompt | 8% | 4% | 3% | 7% | 1% | 8/81 | 2.21 s | 957 | $0 |
| Llama 3.2 1B, untuned, rules + 3 examples | 93% | 53% | 38% | 87% | 26% | 15/81 | 2.62 s | 1,534 | $0 |
| **Llama 3.2 1B, fine-tuned (LoRA)** | **100%** | **96%** | **87%** | **100%** | **84%** | **0/81** | **1.30 s** | **240** | **$0** |
| Gemini 2.5 Flash | 100% | 98% | 70% | 99% | 68% | 2/81 | 1.25 s | 1,002 | $0.48 |
| Gemini 2.5 Flash + thinking | 99% | 99% | 96% | 99% | 96% | 0/81 | 5.43 s | 1,002 | $2.24 |

The fine-tuned model beats Gemini 2.5 Flash on full triage (84% vs 68%) at
the same latency, runs locally with no per-call cost, and missed no real
incidents. Gemini with thinking is more accurate (96%) but about 4× slower.
With 100 alerts, each one is a percentage point, so small gaps are noise.

Merging the adapter into the base weights cut inference latency 6.2% against
the unmerged adapter, matching the base model's speed (1.96 s for 96 tokens on MPS).

Baselines evaluated on the same gold alerts: untuned Llama 3.2 1B (rules
prompt, and rules + 3 examples), Gemini 2.5 Flash (rules prompt, JSON mode,
thinking off and on). Invalid outputs count as wrong on every field and fall
back to `investigate=true`. Hosted-model cost uses list prices passed on the
command line (see `eval_triage.py`); local inference is reported as $0 per call.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login                      # Llama 3.2 is gated: accept the license on Hugging Face first

# Teacher + Gemini baseline via Vertex AI (same setup as ai-operations-center)
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your-project> GOOGLE_CLOUD_LOCATION=us-central1

python generate_data.py --teacher gemini       # dataset -> data/
python review_gold.py --target 200             # build the gold set by hand
export PYTORCH_ENABLE_MPS_FALLBACK=1
python train.py --run_name triage_r8 --rank 8  # ~1-2 h on an M-series Mac
python eval_triage.py --run_name triage_r8     # all baselines on the gold set
python benchmark.py --run_name triage_r8       # merged vs unmerged inference
python make_table.py
```

`python generate_data.py --teacher mock --n 200` runs the whole pipeline offline
for testing (template text, not for real training).

## Limitations

- The alerts are synthetic. They follow the real Orion Commerce services and
  topology, but real alert streams are messier; the gold review guards label
  quality, not realism.
- The teacher model wrote the alert text, which could make the text slightly
  easier for models from the same family. Labels come from code, not the teacher.
- Rules like severity are simplified and documented in `schema.py`.

## Next

- Add the triage node to the ai-operations-center graph and run its eval fixtures end to end
- Quantize the merged model to 4-bit GGUF and serve it through Ollama / llama.cpp
- Expose `triage_alert` as a tool on the trace-incidents MCP server
