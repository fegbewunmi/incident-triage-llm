"""Triage task definition: the Orion Commerce service catalog, the label
taxonomy, the deterministic labeling rules, the prompts, and output parsing.

Everything the model is graded on is derived from rules in this file, so the
ground truth is reproducible and auditable rather than an LLM's opinion.
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

# ── Orion Commerce (mirrors ai-operations-center migrations/0002_seed_orion_commerce.py)

SERVICES = {
    "api-gateway":   {"critical": True,  "team": "Platform", "calls": ["orders", "user-auth"], "external": []},
    "orders":        {"critical": True,  "team": "Commerce", "calls": ["payments", "inventory", "notifications"], "external": []},
    "payments":      {"critical": True,  "team": "Payments", "calls": [], "external": ["stripe"]},
    "inventory":     {"critical": False, "team": "Commerce", "calls": [], "external": []},
    "notifications": {"critical": False, "team": "Platform", "calls": [], "external": ["sendgrid", "twilio"]},
    "user-auth":     {"critical": True,  "team": "Security", "calls": [], "external": ["identity-provider"]},
}
SERVICE_NAMES = list(SERVICES)

# ── Label taxonomy

FailureFamily = Literal[
    "failed_deployment",     # code release to the alerting service shortly before onset
    "config_change",         # config/flag/secret change with no version bump shortly before onset
    "latency_regression",    # gradual latency climb over hours/days, often after a schema/DB change
    "resource_leak",         # memory/connections/threads climbing steadily hours after a release
    "dependency_failure",    # alerting service is healthy itself; a callee or external provider is failing
    "capacity_saturation",   # traffic surge beyond capacity, no relevant change
    "noise",                 # transient blip, flapping, maintenance window, probe/scrape artifact
]
FAMILIES: list[str] = list(FailureFamily.__args__)

Severity = Literal["P1", "P2", "P3"]
Specialist = Literal["telemetry", "deployment", "knowledge"]
Origin = Literal["api-gateway", "orders", "payments", "inventory", "notifications", "user-auth", "external"]

# Which specialist agents the planner should start with, per family.
FIRST_SPECIALISTS: dict[str, list[str]] = {
    "failed_deployment":   ["deployment", "telemetry"],
    "config_change":       ["deployment", "knowledge"],
    "latency_regression":  ["telemetry", "deployment"],
    "resource_leak":       ["telemetry", "deployment"],
    "dependency_failure":  ["telemetry", "knowledge"],
    "capacity_saturation": ["telemetry"],
    "noise":               [],
}


def severity_for(service: str, impact: str, family: str) -> str:
    """P1: critical-path service with high impact.
    P2: critical-path service with medium impact, or non-critical service with high impact.
    P3: everything else, including all noise."""
    if family == "noise":
        return "P3"
    critical = SERVICES[service]["critical"]
    if critical and impact == "high":
        return "P1"
    if (critical and impact == "medium") or (not critical and impact == "high"):
        return "P2"
    return "P3"


class TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    investigate: bool
    failure_family: FailureFamily
    severity: Severity
    origin_service: Origin
    first_specialists: list[Specialist]
    rationale: str


def make_label(family: str, service: str, impact: str, origin: str, rationale: str) -> dict:
    return {
        "investigate": family != "noise",
        "failure_family": family,
        "severity": severity_for(service, impact, family),
        "origin_service": origin,
        "first_specialists": FIRST_SPECIALISTS[family],
        "rationale": rationale,
    }


# ── Prompts

# The fine-tuned model learns the rules from data, so it gets a one-line prompt.
SHORT_SYSTEM = "You are the Orion Commerce alert triage model. Return the triage JSON only."


def _topology_text() -> str:
    lines = []
    for name, s in SERVICES.items():
        calls = ", ".join(s["calls"]) or "none"
        ext = ", ".join(s["external"]) or "none"
        tier = "critical path" if s["critical"] else "non-critical"
        lines.append(f"- {name} ({tier}; calls: {calls}; external providers: {ext})")
    return "\n".join(lines)


# Baselines (untuned Llama, Gemini) need the rules spelled out.
FULL_SYSTEM = f"""You are the alert triage step for Orion Commerce, a 6-service e-commerce platform.
Given one monitoring alert and the list of recent deployments, decide how to triage it.

Services:
{_topology_text()}

failure_family — pick exactly one:
- failed_deployment: a code release (version change) to the alerting service shortly before onset (under ~1 hour), followed by errors or latency.
- config_change: a config-only change (feature flag, env var, secret rotation; version unchanged) to the alerting service shortly before onset.
- latency_regression: latency climbing gradually over hours or days rather than a sudden step, often after a schema or database change days earlier.
- resource_leak: memory, connections, file handles or threads growing steadily, typically hours after a release, possibly with OOM restarts.
- dependency_failure: the alerting service is failing because a service it calls, or an external provider, is failing (timeouts, 5xx, connection refused to that dependency).
- capacity_saturation: request volume far above baseline with CPU or queue saturation, and no relevant recent change.
- noise: not a real incident — a brief blip that self-resolved, a flapping alert, a scheduled maintenance window, a single synthetic probe or metrics-scrape artifact.

Deployments to other services, or deployments long before onset, are usually irrelevant.

severity:
- P1: critical-path service with high impact.
- P2: critical-path service with medium impact, or non-critical service with high impact.
- P3: anything else, and always for noise.
Impact: high = severe customer-facing degradation (e.g. error rate above ~20%, latency 8x+ baseline, OOM kills, traffic 4x+ capacity); medium = clear but partial degradation; low = minor.

origin_service: the service where the problem actually originates. For dependency_failure this is the failing dependency (use "external" for third-party providers); otherwise it is the alerting service.

investigate: false only for noise.

first_specialists by family:
{json.dumps(FIRST_SPECIALISTS)}

rationale: one short sentence citing the evidence.

Return only a JSON object with exactly these keys:
{{"investigate": bool, "failure_family": str, "severity": "P1"|"P2"|"P3", "origin_service": str, "first_specialists": [str], "rationale": str}}
No prose, no markdown."""


def render_alert(alert: dict) -> str:
    """The user message: the alert payload as compact JSON."""
    return json.dumps(alert, ensure_ascii=False, separators=(",", ":"))


def render_label(label: dict) -> str:
    """The assistant target: the triage JSON, keys in a fixed order."""
    ordered = {k: label[k] for k in TriageOutput.model_fields}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def parse_output(text: str) -> tuple[dict | None, str]:
    """Extract and validate a triage object from model text.
    Returns (parsed_dict_or_None, status) where status is ok | bad_json | bad_schema."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None, "bad_json"
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, "bad_json"
    try:
        return TriageOutput(**obj).model_dump(), "ok"
    except (ValidationError, TypeError):
        return None, "bad_schema"
