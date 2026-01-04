#!/usr/bin/env python3
"""Select a deterministic stratified panel of labeled snapshots.

Input: labeled_snapshots.json (list of dicts, each containing 'quadrant' or 'quadrant_label')
Output: panel_200.json (JSON with list of snapshot ids)

Default: N=200, 50 per quadrant (best-effort if some quadrants have fewer).
"""

import argparse
import json
import os
import random
from collections import defaultdict


def _get_quad(x: dict) -> str:
    return x.get("quadrant") or x.get("quadrant_label") or x.get("labels", {}).get("quadrant") or "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.labeled, "r") as f:
        data = json.load(f)

    rng = random.Random(args.seed)

    buckets = defaultdict(list)
    for x in data:
        sid = x.get("id") or x.get("snapshot_id")
        if not sid:
            continue
        buckets[_get_quad(x)].append(sid)

    # Prefer 4-way split if possible
    quads = [q for q in buckets.keys() if q != "UNKNOWN"]
    quads = sorted(quads)
    targets = {}
    if len(quads) >= 4:
        per = args.n // 4
        for q in quads[:4]:
            targets[q] = per
        # remainder goes to the largest bucket
        rem = args.n - per * 4
        if rem > 0:
            biggest = max(quads[:4], key=lambda q: len(buckets[q]))
            targets[biggest] += rem
    else:
        total = sum(len(v) for v in buckets.values())
        for q, ids in buckets.items():
            targets[q] = max(1, int(args.n * len(ids) / max(total, 1)))

    panel = []
    for q, n_q in targets.items():
        ids = list(buckets.get(q, []))
        rng.shuffle(ids)
        panel.extend(ids[:n_q])

    # Trim/pad deterministically
    rng.shuffle(panel)
    panel = panel[:args.n]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"n": len(panel), "seed": args.seed, "ids": panel, "targets": targets}, f, indent=2)

    print(f"Wrote panel of {len(panel)} ids to {args.output}")


if __name__ == "__main__":
    main()
