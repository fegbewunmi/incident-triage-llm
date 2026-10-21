"""Generate the labeled alert dataset.

1. scenarios.py samples N specs with ground-truth labels fixed in code.
2. A teacher LLM writes realistic alert text (name, description, metadata)
   and a one-line rationale for each spec, in batches.
3. Outputs that leak the label or break format are rejected and retried.
4. Results are split into train / val / test by a fixed seed.

Resumable: progress is appended to data/generated.jsonl, and re-running skips
ids already there.

Example:
  python generate_data.py --teacher gemini            # ~3800 alerts, ~10-20 min
  python generate_data.py --teacher mock --n 200      # offline pipeline test
"""

import argparse
import json
import os
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from scenarios import sample_specs
from schema import SERVICES, make_label
from teachers import make_client

WRITER_SYSTEM = """You write realistic production monitoring alerts for Orion Commerce, a 6-service e-commerce platform (api-gateway, orders, payments, inventory, notifications, user-auth).

You receive a JSON list of incident specs. Each spec states the true situation: the alerting service, what is really going on (family), the evidence numbers, and the recent deployments. For EACH spec, write the alert exactly as a monitoring system (Prometheus Alertmanager, Cloud Monitoring, Datadog) and on-call tooling would show it.

Return a JSON array with one object per spec, same order, each with:
- "id": copy from the spec
- "alert_name": a CamelCase alert rule name such as HighErrorRate, P99LatencyHigh, PodMemoryHigh, UpstreamTimeouts, RequestRateAnomaly, ContainerRestarts. Vary it; it must fit the symptom.
- "description": 1-3 sentences in the voice of an alerting system or terse on-call note. Include the evidence numbers (you may round, convert units, or phrase them differently). Vary style a lot: some terse and label-like, some verbose, some with dashboards or query snippets.
- "alert_metadata": an object with 2-5 plausible keys (e.g. region, cluster, pod, namespace, dashboard, runbook, alert_source, environment, labels).
- "rationale": one sentence an expert would write explaining the correct diagnosis from the evidence.

Rules:
- The alert describes SYMPTOMS only. Never state the diagnosis or category in alert_name, description or alert_metadata. Do not use these words there: noise, false positive, leak, failed deployment, dependency failure, config change, capacity, saturation, regression, severity, P1, P2, P3, triage.
- For dependency specs, the symptom shows on the alerting service, with errors/timeouts when it calls the named dependency.
- For noise specs, describe the alert honestly as it would appear (it may mention the blip resolving, flapping, a maintenance window, or a single probe) without calling it noise.
- Do not repeat the recent deployment list in the description; it is attached separately.
- Return only the JSON array."""

# Two checks. Substring match on text with spaces/underscores/hyphens removed and
# lowercased, so CamelCase names like "LatencyRegression" are caught too; plus a
# word-boundary check for P1/P2/P3 (substring would hit things like "p100").
LEAK_SUBSTRINGS = re.compile(
    r"(noise|falsepositive|leak|faileddeploy|dependencyfailure|configchange|capacity|saturat|regression|severity|triage)"
)
LEAK_PRIORITY = re.compile(r"\bp[123]\b", re.IGNORECASE)


def find_leak(text: str) -> str | None:
    m = LEAK_SUBSTRINGS.search(re.sub(r"[\s_\-]", "", text.lower()))
    if m:
        return m.group(0)
    m = LEAK_PRIORITY.search(text)
    return m.group(0) if m else None


def alert_surface(alert_name: str, description: str, metadata: dict) -> str:
    return f"{alert_name} {description} {json.dumps(metadata)}"


def teacher_view(spec: dict) -> dict:
    keep = ["id", "family", "service", "dependency_name", "impact", "evidence", "recent_deployments"]
    return {k: spec[k] for k in keep}


def validate(item: dict, spec: dict) -> str | None:
    """Return an error string, or None if the generated alert is usable."""
    for k in ("alert_name", "description", "alert_metadata", "rationale"):
        if k not in item:
            return f"missing {k}"
    if not isinstance(item["alert_metadata"], dict):
        return "metadata not an object"
    leak = find_leak(alert_surface(item["alert_name"], item["description"], item["alert_metadata"]))
    if leak:
        return f"label leak: {leak!r}"
    if not (10 <= len(item["description"]) <= 600):
        return "description length"
    if not (5 <= len(item["rationale"]) <= 400):
        return "rationale length"
    return None


def build_record(spec: dict, item: dict) -> dict:
    alert = {
        "alert_name": item["alert_name"],
        "service_name": spec["service"],
        "service_tier": "critical" if SERVICES[spec["service"]]["critical"] else "non-critical",
        "onset_timestamp": spec["onset_timestamp"],
        "description": item["description"],
        "alert_metadata": item["alert_metadata"],
        "recent_deployments": spec["recent_deployments"],
    }
    label = make_label(spec["family"], spec["service"], spec["impact"], spec["origin"], item["rationale"].strip())
    return {"id": spec["id"], "alert": alert, "label": label, "spec": spec}


def run_batch(client, batch: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (good_records, specs_to_retry)."""
    user = "Incident specs:\n" + json.dumps([teacher_view(s) for s in batch], indent=1)
    text, _, _ = client.complete(WRITER_SYSTEM, user, json_mode=True, temperature=0.9, max_tokens=8192)
    try:
        items = json.loads(text[text.index("[") : text.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return [], batch
    by_id = {it.get("id"): it for it in items if isinstance(it, dict)}
    good, retry = [], []
    for spec in batch:
        item = by_id.get(spec["id"])
        err = "missing from response" if item is None else validate(item, spec)
        if err:
            retry.append(spec)
        else:
            good.append(build_record(spec, item))
    return good, retry


def split(records: list[dict], val: int, test: int, seed: int) -> dict:
    rs = sorted(records, key=lambda r: r["id"])
    random.Random(seed).shuffle(rs)
    return {"test": rs[:test], "val": rs[test : test + val], "train": rs[test + val :]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", choices=["gemini", "anthropic", "mock"], default="gemini")
    p.add_argument("--model", default=None)
    p.add_argument("--n", type=int, default=3800)
    p.add_argument("--val", type=int, default=300)
    p.add_argument("--test", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_rounds", type=int, default=3, help="retry rounds for rejected specs")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "generated.jsonl")
    done = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["id"]] = r
    specs = [s for s in sample_specs(args.n, args.seed) if s["id"] not in done]
    print(f"{len(done)} already generated, {len(specs)} to go (teacher={args.teacher})")

    client = make_client(args.teacher, args.model, thinking_budget=0)
    lock = threading.Lock()
    rejected_total = 0

    for round_i in range(args.max_rounds):
        if not specs:
            break
        batches = [specs[i : i + args.batch] for i in range(0, len(specs), args.batch)]
        retry_specs = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool, open(progress_path, "a") as out:
            futures = [pool.submit(run_batch, client, b) for b in batches]
            for n_done, fut in enumerate(as_completed(futures), 1):
                good, retry = fut.result()
                with lock:
                    for r in good:
                        out.write(json.dumps(r) + "\n")
                        done[r["id"]] = r
                    out.flush()
                    retry_specs += retry
                if n_done % 25 == 0 or n_done == len(batches):
                    print(f"  round {round_i + 1}: {n_done}/{len(batches)} batches, {len(done)} records")
        rejected_total += len(retry_specs)
        specs = retry_specs
        if specs:
            print(f"  {len(specs)} rejected (leak/format), retrying")
    if specs:
        print(f"dropping {len(specs)} specs that never produced a valid alert")

    records = list(done.values())
    parts = split(records, args.val, args.test, args.seed)
    for name, rows in parts.items():
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        fam = Counter(r["label"]["failure_family"] for r in rows)
        sev = Counter(r["label"]["severity"] for r in rows)
        print(f"{name}: {len(rows)}  families={dict(fam)}  severity={dict(sev)}")
    print(f"total rejected attempts: {rejected_total}")


if __name__ == "__main__":
    main()
