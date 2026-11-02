#!/usr/bin/env python
"""Evaluation at T_max. `predict` runs a backend (unexecuted); `score` and `compare` read prediction
JSONL with fields item_id and text."""
import argparse
import json
from pathlib import Path

import numpy as np

from avrsi.config import load_experiment, load_model
from avrsi.critic import actor_prompt, prompt_digest
from avrsi.data import load_benchmark
from avrsi.metrics import (accuracy, avhbench_score, balanced_accuracy, cmm_score, f1, modality_dominance_index, reasoning_length,
                           swap_sensitivity, xm_balanced_accuracy, xrs, yes_rate)
from avrsi.parse import eval_answer, gold_answer
from avrsi.perturb import load_frozen
from avrsi.stats import accuracy_term, avhbench_term, cmm_term, decide, mcnemar, paired_delta_bootstrap, xm_term


XRS_TERMS = ("cmm", "avhbench", "xm")


def load_items(args):
    if args.benchmark == "xm":
        return [it.to_item(args.root) for it in load_frozen(args.annotations)]
    return load_benchmark(args.benchmark, args.annotations, args.root)


def read_predictions(path):
    return {r["item_id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}


def rows_for(items, predictions):
    rows = []
    for item in items:
        record = predictions[item.item_id]
        pred, gold = eval_answer(record["text"], item), gold_answer(item)
        rows.append({**item.meta, "item_id": item.item_id, "pred": pred, "gold": gold, "correct": pred == gold,
                     "tokens": record.get("tokens", 0)})
    return rows


def score(benchmark, rows):
    preds, golds = [r["pred"] for r in rows], [r["gold"] for r in rows]
    result = {"n": len(rows), "accuracy": accuracy(preds, golds), "balanced_accuracy": balanced_accuracy(preds, golds),
              "f1": f1(preds, golds), "yes_rate": yes_rate(preds), "reasoning_length": reasoning_length([r["tokens"] for r in rows])}
    if benchmark == "cmm":
        result["cmm"] = cmm_score(preds, golds)
    elif benchmark == "avhbench":
        result["avhbench"] = avhbench_score(rows)
    elif benchmark == "xm":
        result |= {"xm_balanced_accuracy": xm_balanced_accuracy(rows), "swap_sensitivity": swap_sensitivity(rows),
                   "modality_dominance_index": modality_dominance_index(rows)}
    return result


def term_for(benchmark, rows):
    golds = [r["gold"] for r in rows]
    if benchmark == "cmm":
        return cmm_term([r["sample_id"] for r in rows], golds)
    if benchmark == "avhbench":
        return avhbench_term([r["task"] for r in rows], golds)
    return xm_term([r["sources"] for r in rows], [r["twin_id"] for r in rows], [r["family"] for r in rows], [r["answer_class"] for r in rows])


def cmd_predict(args):
    from avrsi.models import ACTOR, build_model_input, load_backend
    from avrsi.config import resolve_arm
    from avrsi.train import samplings_from

    experiment = load_experiment()
    model = load_model(args.model)
    backend = load_backend(model)
    arm = resolve_arm("A8")
    if args.adapter:
        backend.attach_lora(arm.lora_r)
        backend.load_adapters(args.adapter, trainable=False)
    greedy = samplings_from(experiment, arm).greedy
    Path(f"{args.out}.meta.json").write_text(json.dumps({"prompt_digest": prompt_digest(), "modalities": args.modalities}))
    with open(args.out, "w") as f:
        for item in load_items(args):
            x = build_model_input(item, args.modalities, actor_prompt(item), model["missing_modality_fill"])
            out = backend.generate([x], greedy, ACTOR)[0]
            f.write(json.dumps({"item_id": item.item_id, "text": out.text, "tokens": len(out.token_ids)}) + "\n")


def cmd_score(args):
    rows = rows_for(load_items(args), read_predictions(args.predictions))
    print(json.dumps(score(args.benchmark, rows), indent=2))


def parse_system(spec):
    """cmm=a.jsonl,b.jsonl: one file per seed."""
    name, files = spec.split("=")
    return name, files.split(",")


def cmd_compare(args):
    annotations = dict(a.split("=") for a in args.annotation)
    roots = dict(a.split("=") for a in args.media_root)
    specs_a, specs_b = dict(map(parse_system, args.a)), dict(map(parse_system, args.b))
    if set(specs_a) != set(specs_b) or len(specs_a) != len(args.a) or len(specs_b) != len(args.b):
        raise SystemExit(f"--a and --b must name the same benchmarks once each: {sorted(specs_a)} vs {sorted(specs_b)}")
    if not args.accuracy and set(specs_a) != set(XRS_TERMS):
        raise SystemExit(f"ΔXRS needs exactly {XRS_TERMS}, got {sorted(specs_a)}")
    terms, a, b = {}, {}, {}
    for name, files_a in specs_a.items():
        files_b = specs_b[name]
        ns = argparse.Namespace(benchmark=name, annotations=annotations[name], root=roots[name])
        items = load_items(ns)
        if name == "avhbench":
            items = [i for i in items if i.meta["task"] in ("a2v", "v2a")]
        rows_a = [rows_for(items, read_predictions(f)) for f in files_a]
        rows_b = [rows_for(items, read_predictions(f)) for f in files_b]
        terms[name] = accuracy_term(None, len(items)) if args.accuracy else term_for(name, rows_a[0])
        a[name] = np.array([[r["correct"] for r in rows] for rows in rows_a], float)
        b[name] = np.array([[r["correct"] for r in rows] for rows in rows_b], float)
    result = paired_delta_bootstrap(terms, a, b, args.bootstraps, args.seed)
    if not args.accuracy:
        result |= {f"xrs_{k}": xrs(*(terms[n].score(v[n]) for n in XRS_TERMS)) for k, v in (("a", a), ("b", b))}
    result["mcnemar_first_seed"] = {n: mcnemar(a[n][0] > 0, b[n][0] > 0) for n in terms}  # secondary, §7.4
    print(json.dumps(result))


def cmd_decide(args):
    """§7.4 rules. JSON: {backbone: {"H1": compare output, "H5": ..., "H3": ..., "H3b": ..., "H4_A5": ...}}."""
    print(json.dumps(decide(json.loads(Path(args.results).read_text()))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("predict", cmd_predict), ("score", cmd_score)):
        p = sub.add_parser(name)
        p.add_argument("--benchmark", required=True)
        p.add_argument("--annotations", required=True)
        p.add_argument("--root", default="")
        p.set_defaults(run=fn)
        if name == "predict":
            p.add_argument("--model", required=True)
            p.add_argument("--adapter")
            p.add_argument("--modalities", nargs="+", choices=["audio", "video"], default=["audio", "video"],
                           help="T9: A-only and V-only baselines")
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--predictions", required=True)
    c = sub.add_parser("compare", help="ΔXRS: paired cluster bootstrap")
    c.add_argument("--a", nargs="+", required=True, help="benchmark=seed0.jsonl,seed1.jsonl")
    c.add_argument("--b", nargs="+", required=True)
    c.add_argument("--annotation", nargs="+", required=True, help="benchmark=path")
    c.add_argument("--media-root", nargs="+", required=True, help="benchmark=path")
    c.add_argument("--bootstraps", type=int, default=10_000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--accuracy", action="store_true", help="plain accuracy per benchmark (H5), not XRS terms")
    c.set_defaults(run=cmd_compare)
    d = sub.add_parser("decide", help="§7.4 decision rules")
    d.add_argument("--results", required=True)
    d.set_defaults(run=cmd_decide)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
