#!/usr/bin/env python
"""Phase 1: Gate 0 on one backbone's planted XM-Gate set, no training. Unexecuted on real models."""
import argparse
import json
from pathlib import Path

from avrsi.config import load_experiment, load_model, resolve_arm
from avrsi.critic import BackendExecutor, JudgeExecutor, prompt_digest
from avrsi.judges import make_judge
from avrsi.judges.media import view_media
from avrsi.metrics import gate0_criteria, gate0_decision, gate0_passed, gate0_report
from avrsi.models import REFERENCE, Sampling, load_backend
from avrsi.perturb import load_planted
from avrsi.sync import attach_auroc, gate0_run, run_probes
from avrsi.train import samplings_from


def run_backbone(args):
    experiment = load_experiment()
    model = load_model(args.model)
    fill = model["missing_modality_fill"]
    backend = load_backend(model)
    samplings = samplings_from(experiment, resolve_arm("A8"))
    planted, base_report = load_planted(args.planted)
    resolve = lambda it: it.to_item(args.media_root)  # noqa: E731
    executor = BackendExecutor(backend, samplings.critic, REFERENCE, fill)
    run = gate0_run(executor, executor, planted, resolve)
    rollout = experiment["sampling"]["rollout"]
    sampled = Sampling(**rollout, max_new_tokens=experiment["sampling"]["critic_max_new_tokens"])
    for mode, results in (("isolated", run["isolated"]), ("joint", run["joint"])):
        attach_auroc(backend, samplings.critic, sampled, results, planted, mode, REFERENCE, fill, resolve,
                     experiment["sampling"]["auroc_sampled_passes"])
    report = gate0_report(run["isolated"], run["joint"], run["order_consistency"], run["pair_parse_failure_rate"])
    report |= {"criteria": gate0_criteria(report), "passed": gate0_passed(report), "probes": run_probes(executor, planted, resolve),
               "base_paths": base_report, "prompt_digest": prompt_digest()}
    if args.gemini_reference:  # upper reference, no pass criterion
        judge = make_judge(experiment["judges"]["gemini"], "gate0")
        media = lambda item, modalities: view_media(item, modalities, Path(args.out).parent / "judge_media")  # noqa: E731
        ref = JudgeExecutor(judge, media)
        reference = gate0_run(ref, ref, planted, resolve)
        report["gemini_reference"] = gate0_report(reference["isolated"], reference["joint"], reference["order_consistency"],
                                                  reference["pair_parse_failure_rate"])
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], **report["criteria"]}))


def run_decide(args):
    reports = {Path(p).stem: json.loads(Path(p).read_text()) for p in args.reports}
    print(json.dumps(gate0_decision(reports)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--planted", required=True, help="gate_planted.json from build_xm.py plant")
    r.add_argument("--media-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--gemini-reference", action="store_true")
    r.set_defaults(run=run_backbone)
    d = sub.add_parser("decide", help="§9 stop rule over per-backbone reports named <backbone>.json")
    d.add_argument("reports", nargs="+")
    d.set_defaults(run=run_decide)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
