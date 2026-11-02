#!/usr/bin/env python
"""Phase 0b. `splits` builds and freezes XM from the clip table; `plant` builds a backbone's planted
sets from θ_0 rollouts; `mux` writes the media. Unexecuted on real data."""
import argparse
import json
import random
from pathlib import Path

from avrsi.config import load_experiment, load_model
from avrsi.perturb import (GATE_PER_TYPE, GUARD_PER_TYPE, assert_source_disjoint, build_planted, build_xm, freeze, label_disjoint_pairs,
                           load_clips, load_frozen, mux_item, overlap_filter, save_planted, select_base_paths, split_sources,
                           verified_clips)


def similarity_table(path):
    return {frozenset(k.split("|")): v for k, v in json.loads(Path(path).read_text()).items()}


def cmd_splits(args):
    exp = load_experiment()["xm"]
    clips = verified_clips(load_clips(args.clips), args.threshold, args.exclude_class)
    known = Path(args.known_ids).read_text().split() if args.known_ids else []
    clips, dropped = overlap_filter(clips, known)
    print(f"overlap filter dropped {len(dropped)} clips")
    disjoint = label_disjoint_pairs(similarity_table(args.similarity), exp["similarity_tier"])
    splits = split_sources(clips, exp["sizes"], args.seed)
    xm = build_xm(splits, disjoint, args.seed, args.threshold, exp["sizes"], exp["silence_share_of_f1"], exp["planted_guard"])
    assert_source_disjoint(xm)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in xm.items():
        print(name, len(items), freeze(items, out / f"{name}.jsonl"))


def cmd_plant(args):
    from avrsi.models import load_backend
    from avrsi.sync import rollout_draws
    from avrsi.train import samplings_from
    from avrsi.config import resolve_arm

    experiment = load_experiment()
    model = load_model(args.model)
    backend = load_backend(model)
    samplings = samplings_from(experiment, resolve_arm("A8"))
    fill = model["missing_modality_fill"]
    disjoint = label_disjoint_pairs(similarity_table(args.similarity), experiment["xm"]["similarity_tier"])
    rng = random.Random(args.seed)
    for name, per_type in (("gate", GATE_PER_TYPE), ("guard", GUARD_PER_TYPE)):
        frozen = load_frozen(Path(args.xm_dir) / f"{name}.jsonl")
        carriers = [it for it in frozen if name == "gate" or it.planted_carrier]
        items = [it.to_item(args.media_root) for it in carriers]
        theta0 = rollout_draws(backend, items, samplings.rollout, args.draws, fill)
        exemplar = None
        if not _enough_valid(theta0):
            exemplar = {k: v[0] for k, v in rollout_draws(backend, items, samplings.rollout, 1, fill, exemplar=True).items()}
        bases, report = select_base_paths(theta0, exemplar)
        planted = build_planted(carriers, bases, disjoint, rng, per_type)
        save_planted(planted, report, Path(args.out) / f"{name}_planted.json")
        print(name, len(planted), report)


def _enough_valid(theta0):
    from avrsi.parse import valid_format
    from avrsi.perturb import MIN_VALID_BASE_RATE

    draws = [t for texts in theta0.values() for t in texts]
    return sum(map(valid_format, draws)) / len(draws) >= MIN_VALID_BASE_RATE


def cmd_mux(args):
    paths = dict(line.split("\t", 1) for line in Path(args.clip_paths).read_text().splitlines())
    Path(args.media_root).mkdir(parents=True, exist_ok=True)
    for name in args.split:
        for item in load_frozen(Path(args.xm_dir) / f"{name}.jsonl"):
            mux_item(item, paths.__getitem__, args.media_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("splits")
    s.add_argument("--clips", required=True, help="CSV, see perturb.load_clips")
    s.add_argument("--similarity", required=True, help='JSON {"classA|classB": similarity}')
    s.add_argument("--threshold", type=float, required=True, help="calibrated SigLIP threshold")
    s.add_argument("--exclude-class", action="append", default=[])
    s.add_argument("--known-ids", help="whitespace-separated clip and YouTube ids of other sets")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(run=cmd_splits)
    p = sub.add_parser("plant")
    p.add_argument("--model", required=True)
    p.add_argument("--xm-dir", required=True)
    p.add_argument("--media-root", required=True)
    p.add_argument("--similarity", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=cmd_plant)
    m = sub.add_parser("mux")
    m.add_argument("--xm-dir", required=True)
    m.add_argument("--clip-paths", required=True, help="TSV clip_id<TAB>path")
    m.add_argument("--media-root", required=True)
    m.add_argument("--split", nargs="+", default=["train", "guard", "dev", "gate", "test"])
    m.set_defaults(run=cmd_mux)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
