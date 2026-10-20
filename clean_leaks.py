"""Remove alerts whose text gives away the label, from every split.

The original leak filter matched whole words only, so CamelCase alert names
such as "PaymentsP99LatencyRegression" slipped through. This re-checks every
record with the stricter filter in generate_data.find_leak and drops the
offenders. Originals are kept in data/pre_clean/. Split membership of the
remaining records is unchanged, so an in-progress gold review stays valid.

Example:
  python clean_leaks.py
"""

import json
import os
import shutil
from collections import Counter

from generate_data import alert_surface, find_leak


def leak_of(rec):
    a = rec["alert"]
    return find_leak(alert_surface(a["alert_name"], a["description"], a["alert_metadata"]))


def main(data_dir="data"):
    backup = os.path.join(data_dir, "pre_clean")
    os.makedirs(backup, exist_ok=True)
    report = {}
    for name in ("train", "val", "test", "gold"):
        path = os.path.join(data_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        if not os.path.exists(os.path.join(backup, f"{name}.jsonl")):
            shutil.copy(path, backup)
        with open(os.path.join(backup, f"{name}.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        kept, words = [], Counter()
        for r in rows:
            w = leak_of(r)
            if w:
                words[w] += 1
            else:
                kept.append(r)
        with open(path, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        report[name] = {"before": len(rows), "after": len(kept), "removed": len(rows) - len(kept), "words": dict(words.most_common())}
        print(f"{name}: {len(rows)} -> {len(kept)}  removed {len(rows) - len(kept)}  {dict(words.most_common(6))}")
    with open(os.path.join(data_dir, "leak_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
