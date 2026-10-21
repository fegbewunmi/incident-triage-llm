"""Enrich every alert with its service tier from the Orion Commerce catalog.

Error analysis showed most severity mistakes came from the model not knowing
which services are business-critical: that fact was never in the input, so it
had to be memorized from service names. In production the triage step would
look the tier up in the services table before calling the model, so we add it
here as "service_tier": "critical" | "non-critical".

Labels and split membership are unchanged, so the gold review stays valid.
Safe to run twice.

Example:
  python add_service_tier.py
"""

import json
import os

from schema import SERVICES


def enrich(alert: dict) -> dict:
    tier = "critical" if SERVICES[alert["service_name"]]["critical"] else "non-critical"
    out = {}
    for k, v in alert.items():
        if k == "service_tier":
            continue
        out[k] = v
        if k == "service_name":
            out["service_tier"] = tier
    return out


def main(data_dir="data"):
    for name in ("train", "val", "test", "gold"):
        path = os.path.join(data_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        for r in rows:
            r["alert"] = enrich(r["alert"])
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} alerts enriched")


if __name__ == "__main__":
    main()
