"""
Summarize <training_folder>/<job>/save_timings.jsonl written with `save.log_timings: true`.

    python scripts/summarize_save_timings.py output/my_job/save_timings.jsonl [more.jsonl ...]

`save_call` rows are measured on the training thread (how long training was blocked by
each save). `writer` rows are measured on the background writer (async_save only).
"""
import json
import sys
from collections import defaultdict


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float('nan')
    idx = min(len(values) - 1, max(0, int(round(q / 100 * (len(values) - 1)))))
    return values[idx]


def summarize(path):
    groups = defaultdict(lambda: defaultdict(list))
    failed = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('state') == 'FAILED':
                failed += 1
                continue
            key = f"{record.get('kind', 'save_call')}/{record.get('mode')}"
            for name, value in record.items():
                if name.endswith('_ms') and isinstance(value, (int, float)):
                    groups[key][name].append(value)

    print(f"== {path}")
    for key in sorted(groups):
        print(f"  [{key}]")
        for name in sorted(groups[key]):
            values = groups[key][name]
            mean = sum(values) / len(values)
            print(
                f"    {name:<20} n={len(values):<5} mean={mean:10.1f}  "
                f"p95={percentile(values, 95):10.1f}  max={max(values):10.1f}"
            )
    if failed:
        print(f"  FAILED writes: {failed}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        summarize(p)
