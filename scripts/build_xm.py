#!/usr/bin/env python
"""Phase 0b, in order: `similarity` writes the class table; `splits` writes XM drafts from the clip table; `mux` writes the media; `plant`
builds a backbone's planted sets from θ_0 rollouts (dev first, for prompt iteration, then gate and
guard) and freezes them; `freeze` runs the human check and only then freezes the splits. Unexecuted
on real data."""
import argparse
import json
import random
from pathlib import Path

from avrsi.config import load_experiment, load_model, resolve_arm
from avrsi.critic import prompt_digest
from avrsi.perturb import (accept_class_precision, assert_source_disjoint, build_planted, build_xm, collect_base_paths, freeze,
                           human_check_passes, label_disjoint_pairs, load_clips, load_draft, mux_item, overlap_filter, save_planted,
                           split_sources, verified_clips, write_draft)

SPLITS = ("train", "guard", "dev", "gate", "test")


def similarity_table(path):
    return {frozenset(k.split("|")): v for k, v in json.loads(Path(path).read_text()).items()}


def cmd_splits(args):
    exp = load_experiment()["xm"]
    precision = json.loads(Path(args.class_precision).read_text())
    excluded = [c for c, p in precision.items() if not accept_class_precision(p, exp["siglip_min_precision"])]
    print(f"SigLIP precision excluded {len(excluded)} classes")
    clips = verified_clips(load_clips(args.clips), args.threshold, excluded)
    unmeasured = {c.label for c in clips} - set(precision)
    if unmeasured:
        raise SystemExit(f"no measured precision for {sorted(unmeasured)}")
    clips, dropped = overlap_filter(clips, Path(args.known_ids).read_text().split())
    print(f"overlap filter dropped {len(dropped)} clips")
    disjoint = label_disjoint_pairs(similarity_table(args.similarity), exp["similarity_tier"])
    splits = split_sources(clips, exp["sizes"], args.seed)
    xm = build_xm(splits, disjoint, args.seed, args.threshold, exp["sizes"], exp["silence_share_of_f1"], exp["planted_guard"], exp["family_shares"])
    assert_source_disjoint(xm)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in xm.items():
        write_draft(items, out / f"{name}.jsonl")
        print(name, len(items))


def cmd_similarity(args):
    from avrsi.labels import SiglipScorer, clap_text_embeddings, class_similarity, write_similarity

    exp = load_experiment()["xm"]
    classes = Path(args.classes).read_text().splitlines()
    write_similarity(class_similarity(clap_text_embeddings(exp["clap_model"], classes), SiglipScorer(exp["siglip_model"]).text_embeddings(classes)), args.out)


def carriers_for(name, items):
    if name == "dev":
        return [it for it in items if it.family in ("F1", "F2") and it.kind != "silence"]
    return [it for it in items if name == "gate" or it.planted_carrier]


def cmd_plant(args):
    from avrsi.models import load_backend
    from avrsi.sync import rollout_draws
    from avrsi.train import samplings_from

    experiment = load_experiment()
    xm = experiment["xm"]
    model = load_model(args.model)
    backend = load_backend(model)
    rollout = samplings_from(experiment, resolve_arm("A8")).rollout
    fill = model["missing_modality_fill"]
    disjoint = label_disjoint_pairs(similarity_table(args.similarity), xm["similarity_tier"])
    rng = random.Random(args.seed)
    per_type = {"gate": xm["planted_gate"] // 4, "guard": xm["planted_guard"] // 4, "dev": xm["planted_dev_per_type"]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name in args.split:
        carriers = carriers_for(name, load_draft(Path(args.xm_dir) / f"{name}.jsonl"))
        resolved = {it.item_id: it.to_item(args.media_root) for it in carriers}

        def draw(items):
            return rollout_draws(backend, [resolved[it.item_id] for it in items], rollout, args.draws, fill)

        def exemplar_draw(items):
            drawn = rollout_draws(backend, [resolved[it.item_id] for it in items], rollout, 1, fill, exemplar=True)
            return {k: v[0] for k, v in drawn.items()}

        bases, report = collect_base_paths(carriers, draw, exemplar_draw, args.redraws)
        planted = build_planted(carriers, bases, disjoint, rng, per_type[name])
        report["prompt_digest"] = prompt_digest()
        digest = save_planted(planted, report, Path(args.out) / f"{name}_planted.json")
        print(name, len(planted), digest, report)


def cmd_freeze(args):
    """Phase 0b steps 7 and 8. Over the limit means rebuild, so nothing is written."""
    max_error = load_experiment()["xm"]["human_error_max"]
    if not human_check_passes(args.errors, args.checked, max_error):
        raise SystemExit(f"human check failed: {args.errors}/{args.checked} over {max_error}. Rebuild.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        print(name, freeze(load_draft(Path(args.xm_dir) / f"{name}.jsonl"), out / f"{name}.jsonl"))
    print(f"human error rate {args.errors / args.checked:.4f}")


def cmd_mux(args):
    paths = dict(line.split("\t", 1) for line in Path(args.clip_paths).read_text().splitlines())
    Path(args.media_root).mkdir(parents=True, exist_ok=True)
    for name in args.split:
        for item in load_draft(Path(args.xm_dir) / f"{name}.jsonl"):
            mux_item(item, paths.__getitem__, args.media_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("splits")
    s.add_argument("--clips", required=True, help="CSV, see perturb.load_clips")
    s.add_argument("--similarity", required=True, help='JSON {"classA|classB": similarity}')
    s.add_argument("--threshold", type=float, required=True, help="calibrated SigLIP threshold")
    s.add_argument("--class-precision", required=True, help='JSON {"class": measured SigLIP precision}')
    s.add_argument("--known-ids", required=True, help="whitespace-separated clip and YouTube ids of other sets")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(run=cmd_splits)
    c = sub.add_parser("similarity", help="class similarity for --similarity; untested on real models")
    c.add_argument("--classes", required=True, help="one class per line")
    c.add_argument("--out", required=True)
    c.set_defaults(run=cmd_similarity)
    p = sub.add_parser("plant")
    p.add_argument("--model", required=True)
    p.add_argument("--xm-dir", required=True)
    p.add_argument("--media-root", required=True)
    p.add_argument("--similarity", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--redraws", type=int, default=5, help="rounds for items with no valid draw")
    p.add_argument("--split", nargs="+", choices=["dev", "gate", "guard"], default=["gate", "guard"])
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=cmd_plant)
    f = sub.add_parser("freeze")
    f.add_argument("--xm-dir", required=True)
    f.add_argument("--errors", type=int, required=True, help="errors found by the human check")
    f.add_argument("--checked", type=int, default=500)
    f.add_argument("--out", required=True)
    f.set_defaults(run=cmd_freeze)
    m = sub.add_parser("mux")
    m.add_argument("--xm-dir", required=True)
    m.add_argument("--clip-paths", required=True, help="TSV clip_id<TAB>path")
    m.add_argument("--media-root", required=True)
    m.add_argument("--split", nargs="+", default=list(SPLITS))
    m.set_defaults(run=cmd_mux)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
