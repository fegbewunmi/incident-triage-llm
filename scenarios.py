"""Programmatic scenario specs. Each spec fixes the ground truth (family,
alerting service, origin, impact, severity) and the concrete evidence (metric
numbers, deployment history) BEFORE any LLM is involved. The teacher model only
turns a spec into realistic alert text; it never decides the label.

Deployment history is built here too, including distractors (deploys to other
services, or to the same service long before onset) so the model can't learn
the shortcut "any recent deploy means failed_deployment".
"""

import random
from datetime import datetime, timedelta, timezone

from schema import FAMILIES, SERVICE_NAMES, SERVICES, severity_for

FAMILY_WEIGHTS = {
    "noise": 0.20,
    "failed_deployment": 0.17,
    "dependency_failure": 0.15,
    "latency_regression": 0.13,
    "resource_leak": 0.12,
    "capacity_saturation": 0.12,
    "config_change": 0.11,
}
assert set(FAMILY_WEIGHTS) == set(FAMILIES)

CODE_CHANGES = [
    "Upgraded {lib} from {va} to {vb}", "Refactored {area} handler", "Added retry with exponential backoff to {area} client",
    "Switched {area} serializer to orjson", "Bumped base image to python:3.12-slim", "Enabled HTTP/2 on {area} client",
    "Rewrote {area} query to use batch fetch", "Added request validation middleware", "Changed default timeout for {area} calls",
]
CONFIG_CHANGES = [
    "Feature flag {flag} enabled for 100% of traffic", "Rotated {secret}", "Changed {env} from {a} to {b}",
    "Reduced connection pool max from {a} to {b}", "Updated rate limit to {a} req/s", "Switched region routing weight to {a}%",
]
MIGRATIONS = [
    "Dropped index idx_{table}_{col}", "Added column {col} to {table} (backfill running)",
    "Changed {table}.{col} type to TEXT", "Partitioned {table} by created_at",
]
LIBS = ["stripe-python", "sqlalchemy", "httpx", "grpcio", "redis-py", "pydantic", "boto3", "psycopg"]
SERVICE_AREAS = {
    "api-gateway": ["routing", "rate-limiter", "tls", "request-auth"],
    "orders": ["checkout", "cart", "order-state", "fulfilment"],
    "payments": ["charge", "refund", "stripe-webhook", "capture"],
    "inventory": ["stock-sync", "reservation", "sku-lookup"],
    "notifications": ["email", "sms", "template", "queue-consumer"],
    "user-auth": ["login", "token", "session", "sso"],
}
FLAGS = ["checkout_v2", "new_pricing_engine", "async_capture", "sso_redirect", "batch_reservations", "smart_retry"]
SECRETS = ["JWT signing key", "Stripe API key", "database password", "SendGrid API key", "OAuth client secret"]
ENVS = ["MAX_WORKERS", "DB_POOL_SIZE", "CACHE_TTL_SECONDS", "UPSTREAM_TIMEOUT_MS", "GUNICORN_THREADS"]
TABLES = ["orders", "payments", "users", "sessions", "stock_levels", "order_items", "refunds"]
COLS = ["customer_id", "status", "created_at", "sku", "email", "order_id"]
NOISE_KINDS = [
    "brief spike that self-resolved within {n} seconds",
    "alert flapping: fired and cleared {n} times in the last 30 minutes with no sustained impact",
    "fired during a scheduled maintenance window",
    "single synthetic probe from one region failed; real user traffic unaffected",
    "metrics scrape gap made the rate calculation spike; no change in real traffic",
    "threshold breached by one data point at a low-traffic hour; next sample normal",
]


def _fill(rng: random.Random, template: str, service: str = "orders") -> str:
    return template.format(
        lib=rng.choice(LIBS), a=rng.randint(2, 40), b=rng.randint(2, 40), area=rng.choice(SERVICE_AREAS[service]),
        flag=rng.choice(FLAGS), secret=rng.choice(SECRETS), env=rng.choice(ENVS),
        table=rng.choice(TABLES), col=rng.choice(COLS), n=rng.randint(30, 180),
        va=f"{(m := rng.randint(1, 9))}.{(k := rng.randint(0, 20))}.0", vb=f"{m}.{k + 1}.0",
    )


def _version(rng: random.Random) -> tuple[str, str]:
    major, minor, patch = rng.randint(1, 4), rng.randint(0, 12), rng.randint(0, 9)
    return f"v{major}.{minor}.{patch}", f"v{major}.{minor}.{patch + 1}"


def _deploy(rng, service, minutes_before, kind="code") -> dict:
    v_from, v_to = _version(rng)
    if kind == "code":
        changes = [_fill(rng, t, service) for t in rng.sample(CODE_CHANGES, rng.randint(1, 3))]
    elif kind == "config":
        v_to = v_from  # config-only rollout: no version bump
        changes = [_fill(rng, rng.choice(CONFIG_CHANGES), service)]
    else:  # migration
        changes = [_fill(rng, rng.choice(MIGRATIONS), service)]
    return {
        "service": service,
        "version_from": v_from,
        "version_to": v_to,
        "minutes_before_onset": int(minutes_before),
        "change_summary": changes,
    }


def _distractors(rng, exclude: set[str], k_max=2) -> list[dict]:
    pool = [s for s in SERVICE_NAMES if s not in exclude]
    out = []
    for s in rng.sample(pool, rng.randint(0, min(k_max, len(pool)))):
        out.append(_deploy(rng, s, rng.choice([rng.randint(5, 90), rng.randint(180, 4000)]), rng.choice(["code", "config"])))
    return out


def _impact_numbers(rng, family: str, impact: str) -> dict:
    """Concrete evidence whose magnitude matches the impact level."""
    if family in ("failed_deployment", "config_change", "dependency_failure"):
        lo, hi = {"high": (20, 65), "medium": (5, 20), "low": (1, 5)}[impact]
        return {"error_rate_pct": round(rng.uniform(lo, hi), 1), "baseline_error_rate_pct": round(rng.uniform(0.1, 0.6), 1)}
    if family == "latency_regression":
        lo, hi = {"high": (8, 15), "medium": (3, 8), "low": (1.5, 3)}[impact]
        base = rng.randint(60, 300)
        return {"p99_latency_ms": int(base * rng.uniform(lo, hi)), "baseline_p99_ms": base, "climb_duration_hours": rng.randint(3, 60)}
    if family == "resource_leak":
        resource = rng.choice(["memory", "db_connections", "open_file_descriptors", "threads"])
        if impact == "high":
            return {"resource": resource, "utilization_pct": rng.randint(96, 100), "oom_or_restart_count": rng.randint(2, 12)}
        lo, hi = {"medium": (86, 95), "low": (72, 85)}[impact]
        return {"resource": resource, "utilization_pct": rng.randint(lo, hi), "oom_or_restart_count": 0}
    if family == "capacity_saturation":
        lo, hi = {"high": (4, 7), "medium": (2.5, 4), "low": (1.5, 2.5)}[impact]
        return {"request_rate_multiple_of_baseline": round(rng.uniform(lo, hi), 1), "cpu_pct": rng.randint(75, 99),
                "trigger": rng.choice(["flash sale", "marketing email blast", "bot traffic", "partner integration launch", "viral social post", "unknown"])}
    # noise
    return {"kind": _fill(rng, rng.choice(NOISE_KINDS)), "peak_value_note": rng.choice(["error rate", "p99 latency", "5xx count", "queue depth"])}


def sample_spec(rng: random.Random, idx: int) -> dict:
    family = rng.choices(list(FAMILY_WEIGHTS), weights=list(FAMILY_WEIGHTS.values()))[0]

    if family == "dependency_failure":
        candidates = [(s, c) for s in SERVICE_NAMES for c in SERVICES[s]["calls"]] + \
                     [(s, "external") for s in SERVICE_NAMES if SERVICES[s]["external"]]
        service, origin = rng.choice(candidates)
        dependency_name = rng.choice(SERVICES[service]["external"]) if origin == "external" else origin
    else:
        service = rng.choice(SERVICE_NAMES)
        origin, dependency_name = service, None

    impact = "low" if family == "noise" else rng.choices(["high", "medium", "low"], weights=[0.4, 0.4, 0.2])[0]

    deploys = []
    if family == "failed_deployment":
        deploys.append(_deploy(rng, service, rng.randint(4, 50), "code"))
    elif family == "config_change":
        deploys.append(_deploy(rng, service, rng.randint(3, 60), "config"))
    elif family == "resource_leak":
        deploys.append(_deploy(rng, service, rng.randint(120, 900), "code"))
    elif family == "latency_regression" and rng.random() < 0.6:
        deploys.append(_deploy(rng, service, rng.randint(1440, 5000), "migration"))
    elif family in ("capacity_saturation", "noise") and rng.random() < 0.3:
        # same-service deploy, but long before onset: should be ignored
        deploys.append(_deploy(rng, service, rng.randint(1500, 8000), "code"))
    exclude = {service} | ({origin} if origin not in ("external", service) else set())
    deploys += _distractors(rng, exclude)
    rng.shuffle(deploys)

    onset = datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(minutes=rng.randint(0, 60 * 24 * 200))
    return {
        "id": f"T{idx:05d}",
        "family": family,
        "service": service,
        "origin": origin,
        "dependency_name": dependency_name,
        "impact": impact,
        "severity": severity_for(service, impact, family),
        "onset_timestamp": onset.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evidence": _impact_numbers(rng, family, impact),
        "recent_deployments": deploys,
    }


def sample_specs(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    return [sample_spec(rng, i) for i in range(n)]
