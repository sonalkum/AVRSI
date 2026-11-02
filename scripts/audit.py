#!/usr/bin/env python
"""Phase 5 blind audit, AVRSI vs a baseline. Plans by default; only --execute calls a judge, capped
and cached by config. Unexecuted."""
import argparse
import json
import random
from pathlib import Path

from avrsi.config import load_experiment
from avrsi.judges import make_judge
from avrsi.judges.audit import audit_pair
from avrsi.judges.media import view_media
from avrsi.metrics import audit_preference, cohen_kappa
from avrsi.perturb import load_frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True, help="frozen XM JSONL")
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--ours", required=True, help="prediction JSONL")
    parser.add_argument("--baseline", required=True, help="prediction JSONL")
    parser.add_argument("--judges", nargs="+", default=["gemini"], choices=["gemini", "openai"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--work-dir", default="outputs/audit_media")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="make live judge calls")
    args = parser.parse_args(argv)

    read = lambda p: {r["item_id"]: r["text"] for r in map(json.loads, Path(p).read_text().splitlines())}  # noqa: E731
    ours, baseline = read(args.ours), read(args.baseline)
    items = [it.to_item(args.media_root) for it in load_frozen(args.items)]
    random.Random(args.seed).shuffle(items)
    items = [it for it in items if it.item_id in ours and it.item_id in baseline][: args.limit]
    print(f"planned: {len(items)} items × {len(args.judges)} judges = {len(items) * len(args.judges)} calls")
    if not args.execute:
        return
    config = load_experiment()["judges"]
    wins = {}
    for name in args.judges:
        judge, rng = make_judge(config[name]), random.Random(args.seed)
        wins[name] = [audit_pair(judge, it, ours[it.item_id], baseline[it.item_id],
                                 view_media(it, it.modalities, args.work_dir), rng) for it in items]
    result = {f"preference_{n}": audit_preference([w for w in v if w is not None]) for n, v in wins.items()}
    if len(args.judges) == 2:
        a, b = wins[args.judges[0]], wins[args.judges[1]]
        kept = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        result["kappa"] = cohen_kappa(*zip(*kept))
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
