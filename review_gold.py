"""Human review of the test split to build the gold evaluation set.

For each test alert you see what the model will see (the alert) and the label
the scenario spec assigned. Check that the alert text actually supports that
label, then accept, fix, or reject it. Progress is saved after every decision,
so you can stop and resume any time.

Keys:
  Enter  accept as is
  f      change failure_family        s  change severity
  o      change origin_service        x  reject (alert text is wrong or ambiguous)
  q      save and quit

Example:
  python review_gold.py --target 200
"""

import argparse
import json
import os

from schema import FAMILIES, FIRST_SPECIALISTS, SERVICE_NAMES

ORIGINS = SERVICE_NAMES + ["external"]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def choose(prompt, options):
    for i, o in enumerate(options, 1):
        print(f"   {i}. {o}")
    while True:
        raw = input(f"{prompt} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]


def show(rec, i, total):
    a, l = rec["alert"], rec["label"]
    print("\n" + "=" * 78)
    print(f"[{i}/{total}] {rec['id']}")
    print(f"ALERT  {a['alert_name']} on {a['service_name']} at {a['onset_timestamp']}")
    print(f"       {a['description']}")
    print(f"       metadata: {json.dumps(a['alert_metadata'])}")
    if a["recent_deployments"]:
        for d in a["recent_deployments"]:
            print(f"       deploy: {d['service']} {d['version_from']}->{d['version_to']} "
                  f"{d['minutes_before_onset']} min before: {'; '.join(d['change_summary'])}")
    else:
        print("       deploy: none")
    print("-" * 78)
    print(f"LABEL  family={l['failure_family']}  severity={l['severity']}  origin={l['origin_service']}  "
          f"investigate={l['investigate']}")
    print(f"       rationale: {l['rationale']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--target", type=int, default=200, help="stop after this many accepted alerts")
    args = p.parse_args()

    test = load(os.path.join(args.data_dir, "test.jsonl"))
    gold_path = os.path.join(args.data_dir, "gold.jsonl")
    log_path = os.path.join(args.data_dir, "review_log.jsonl")
    gold = load(gold_path)
    reviewed = {r["id"] for r in load(log_path)}
    print(f"{len(gold)} accepted so far, {len(reviewed)} reviewed, target {args.target}")

    for i, rec in enumerate(test, 1):
        if len(gold) >= args.target:
            print("target reached")
            break
        if rec["id"] in reviewed:
            continue
        show(rec, i, len(test))
        label, decision, changes = dict(rec["label"]), None, {}
        while decision is None:
            key = input("Enter=accept  f/s/o=edit  x=reject  q=quit > ").strip().lower()
            if key == "":
                decision = "accept"
            elif key == "f":
                label["failure_family"] = choose("family", FAMILIES)
                label["investigate"] = label["failure_family"] != "noise"
                label["first_specialists"] = FIRST_SPECIALISTS[label["failure_family"]]
                if label["failure_family"] == "noise":
                    label["severity"] = "P3"
                changes["failure_family"] = label["failure_family"]
                print(f"   now: family={label['failure_family']} severity={label['severity']} origin={label['origin_service']}")
            elif key == "s":
                label["severity"] = choose("severity", ["P1", "P2", "P3"])
                changes["severity"] = label["severity"]
            elif key == "o":
                label["origin_service"] = choose("origin", ORIGINS)
                changes["origin_service"] = label["origin_service"]
            elif key == "x":
                decision = "reject"
            elif key == "q":
                print(f"saved. {len(gold)} accepted.")
                return
        if changes and decision == "accept":
            decision = "accept_edited"
        if decision.startswith("accept"):
            gold.append({**rec, "label": label})
            with open(gold_path, "a") as f:
                f.write(json.dumps({**rec, "label": label}) + "\n")
        with open(log_path, "a") as f:
            f.write(json.dumps({"id": rec["id"], "decision": decision, "changes": changes}) + "\n")
        reviewed.add(rec["id"])

    log = load(log_path)
    n_edit = sum(1 for r in log if r["decision"] == "accept_edited")
    n_rej = sum(1 for r in log if r["decision"] == "reject")
    print(f"\nreviewed {len(log)}: accepted {len(gold)} ({n_edit} edited), rejected {n_rej}")


if __name__ == "__main__":
    main()
