"""Training loop on a Backend: rollouts, critic passes, rewards, GRPO, sync (PLAN §3)."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch

from .config import Arm, load_experiment, load_model, resolve_arm
from .critic import BackendExecutor, Executor, GroupVerdicts, actor_prompt, score_group
from .grpo import grpo_loss
from .metrics import fusion_token_share, supported_rate, tie_rate, xm_balanced_accuracy
from .models.base import ACTOR, CRITIC, REFERENCE, Backend, Generation, Sampling, build_model_input
from .parse import Item, ParsedPath, eval_answer, gold_answer, normalise_tokens, parse_path, r_format
from .perturb import sample_batch
from .rewards import group_advantages, pairwise_r, r_outcome, r_pair, r_total, r_total_rl_comm, r_xm
from .sync import Guard, GuardSet, actor_outcome, evaluate_actor


@dataclass
class Samplings:
    rollout: Sampling
    greedy: Sampling
    critic: Sampling


def samplings_from(experiment: dict, arm: Arm) -> Samplings:
    s = experiment["sampling"]
    return Samplings(Sampling(**s["rollout"], max_new_tokens=arm.max_new_tokens),
                     Sampling(**s["greedy"], max_new_tokens=arm.max_new_tokens),
                     Sampling(**s["greedy"], max_new_tokens=s["critic_max_new_tokens"]))


@dataclass
class TrainResult:
    steps: list[dict] = field(default_factory=list)
    dev: list[dict] = field(default_factory=list)
    guard: Guard | None = None


def group_rewards(arm: Arm, item: Item, gens: Sequence[Generation], executor: Executor | None, rng: random.Random,
                  rl_comm_scorer: Callable | None = None) -> tuple[list[float], dict]:
    paths = [parse_path(g.text, item) for g in gens]
    outcome = [r_outcome(p, item) for p in paths]
    fmt = [r_format(p.raw) for p in paths]
    if arm.reward == "rl_comm":
        if rl_comm_scorer is None:
            raise NotImplementedError("A7 needs r_arr and r_avc: PLAN §8 names them but gives no definition, so pass a scorer")
        extra = rl_comm_scorer(item, paths)
        totals = [r_total_rl_comm(o, f, arr, avc, arm.lam) for o, f, (arr, avc) in zip(outcome, fmt, extra)]
        return totals, {"passes": 0, "verdicts": None, "paths": paths}
    verdicts = None
    pair, xm = [0.0] * len(paths), [0.0] * len(paths)
    if arm.critic != "none":
        verdicts = score_group(arm, executor, item, paths, rng)
        if arm.pair_orders != "none":
            pair = r_pair(pairwise_r([p.valid for p in paths], verdicts.pair_outcomes))
        if arm.check_view != "none":
            xm = [r_xm(a, v, item.silence) for a, v in zip(verdicts.audio, verdicts.visual)]
    totals = [r_total(o, f, p, x, arm.lam, arm.alpha, arm.gamma) for o, f, p, x in zip(outcome, fmt, pair, xm)]
    return totals, {"passes": verdicts.passes if verdicts else 0, "verdicts": verdicts, "paths": paths}


def _health(info: Sequence[dict]) -> dict:
    verdicts: list[GroupVerdicts] = [i["verdicts"] for i in info if i["verdicts"] is not None]
    paths: list[ParsedPath] = [p for i in info for p in i["paths"]]
    pairs = [v for g in verdicts for v in g.pair_outcomes.values()]
    claims = [v for g in verdicts for segment in (g.audio, g.visual) for path in segment for v in path]
    valid = [p for p in paths if p.valid]
    shares = [fusion_token_share(p.fusion, len(normalise_tokens(" ".join(p.audio + p.visual + p.fusion)))) for p in valid]
    return {
        "tie_rate": tie_rate(pairs) if pairs else None,
        "supported_rate": supported_rate(claims) if claims else None,
        "fusion_token_share": sum(shares) / len(shares) if shares else None,
        "malformed_rate": 1 - len(valid) / len(paths),
        "check_failures": sum(g.check_failures for g in verdicts),
        "pair_failures": sum(g.pair_failures for g in verdicts),
    }


def grpo_update(backend: Backend, optimizer, rollouts: Sequence[tuple], beta: float) -> dict:
    """One update. Trajectories are backpropagated one at a time; the sum equals the batch mean."""
    optimizer.zero_grad()
    total_loss, total_kl, n = 0.0, 0.0, len(rollouts)
    for model_input, gen, advantage in rollouts:
        logps = backend.score_logprobs([model_input], [gen], ACTOR)[0]
        if logps.numel() == 0:
            continue
        with torch.no_grad():
            ref = backend.score_logprobs([model_input], [gen], REFERENCE)[0]
        mask = torch.ones_like(logps)[None]
        loss, stats = grpo_loss(logps[None], logps.detach()[None], ref[None], torch.tensor([advantage], dtype=logps.dtype), mask, beta)
        (loss / n).backward()
        total_loss += float(loss.detach()) / n
        total_kl += stats["kl"] / n
    optimizer.step()
    return {"loss": total_loss, "kl": total_kl}


def train(arm: Arm, backend: Backend, natural: Sequence[Item], xm_train: Sequence[Item], guard_set: GuardSet | None,
          samplings: Samplings, dev_score: Callable[[Backend, int], dict] | None = None, seed: int = 0, steps: int | None = None,
          out_dir: str | Path | None = None, external_executor: Executor | None = None,
          rl_comm_scorer: Callable | None = None, fill: str | None = None) -> TrainResult:
    if not arm.trained or arm.retired:
        raise ValueError(f"{arm.id} is not a training run")
    if arm.critic == "gemini" and external_executor is None:
        raise ValueError(f"{arm.id} needs an external Gemini executor")
    if arm.guard == "xm_guard" and guard_set is None:
        raise ValueError(f"{arm.id} needs XM-Guard")
    steps = arm.T_max if steps is None else steps
    rng = random.Random(seed)
    torch.manual_seed(seed)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    executor = external_executor if arm.critic == "gemini" else BackendExecutor(backend, samplings.critic, CRITIC, fill)
    if arm.critic == "none":
        executor = None
    guard = Guard(arm, out / "guard.jsonl" if out else None) if arm.has_handoff else None
    if arm.guard == "xm_guard":
        guard.set_reference(evaluate_actor(backend, guard_set, arm, actor_prompt, samplings.greedy, samplings.critic, fill, reference=True))
    optimizer = torch.optim.AdamW(backend.trainable_parameters(), lr=arm.lr)
    result = TrainResult(guard=guard)
    for step in range(1, steps + 1):
        started = time.time()
        batch = sample_batch(natural, xm_train, arm.B, arm.p, rng)
        rollouts, info, totals_all = [], [], []
        for item in batch:
            model_input = build_model_input(item, item.modalities, actor_prompt(item), fill)
            gens = backend.generate([model_input] * arm.K, samplings.rollout, ACTOR)
            totals, details = group_rewards(arm, item, gens, executor, rng, rl_comm_scorer)
            advantages = group_advantages(totals)
            rollouts += [(model_input, g, a) for g, a in zip(gens, advantages)]
            info.append(details)
            totals_all.append(totals)
        update = grpo_update(backend, optimizer, rollouts, arm.beta)
        record = {
            "step": step, **update, **_health(info),
            "reward_mean": sum(map(sum, totals_all)) / sum(map(len, totals_all)),
            "reward_group_spread": max(max(t) - min(t) for t in totals_all),
            "passes": sum(i["passes"] for i in info),
            "tokens_mean": sum(len(g.token_ids) for _, g, _ in rollouts) / len(rollouts),
            "step_seconds": time.time() - started,
        }
        if arm.has_handoff and arm.T and step % arm.T == 0:
            record["handoff"] = _sync(arm, backend, guard, guard_set, batch, samplings, step, fill)
        if dev_score and step % arm.dev_every == 0:
            result.dev.append({"step": step, **dev_score(backend, step)})
        result.steps.append(record)
        if out:
            with (out / "steps.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
    return result


def _sync(arm: Arm, backend: Backend, guard: Guard, guard_set, batch, samplings: Samplings, step: int, fill) -> bool:
    if arm.guard == "minibatch":
        actor, _ = actor_outcome(backend, batch, actor_prompt, samplings.greedy, fill, role=ACTOR)
        critic, _ = actor_outcome(backend, batch, actor_prompt, samplings.greedy, fill, role=CRITIC)
        decision = guard.decide_minibatch(step, actor, critic)
    else:
        scores = evaluate_actor(backend, guard_set, arm, actor_prompt, samplings.greedy, samplings.critic, fill)
        decision = guard.decide(step, scores)
    if decision.handoff:
        backend.sync_critic()
    return decision.handoff


def xm_dev_scorer(items: Sequence[Item], samplings: Samplings, fill: str | None = None) -> Callable[[Backend, int], dict]:
    """§3.7 dev checkpoints: balanced accuracy on XM-Dev, greedy, with the evaluation extractor."""
    def score(backend: Backend, _step: int) -> dict:
        inputs = [build_model_input(it, it.modalities, actor_prompt(it), fill) for it in items]
        outputs = backend.generate(inputs, samplings.greedy, ACTOR)
        rows = [{**it.meta, "correct": eval_answer(o.text, it) == gold_answer(it)} for it, o in zip(items, outputs)]
        return {"xm_dev_balanced_accuracy": xm_balanced_accuracy(rows)}
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one AVRSI arm (untested on real models)")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True, help="configs/models/<name>.yaml")
    parser.add_argument("--natural", required=True, help="benchmark registry name for natural training data")
    parser.add_argument("--natural-root", required=True)
    parser.add_argument("--xm-dir", required=True, help="directory with frozen train.jsonl, guard.jsonl, dev.jsonl")
    parser.add_argument("--planted", required=True, help="this backbone's guard_planted.json from build_xm.py plant")
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--tuned", default="{}", help="JSON of tuned values")
    args = parser.parse_args()

    from .data import load_benchmark
    from .models import load_backend
    from .perturb import load_frozen, load_planted

    experiment = load_experiment()
    arm = resolve_arm(args.arm, json.loads(args.tuned), experiment)
    model = load_model(args.model)
    backend = load_backend(model)
    if arm.trained:
        backend.attach_lora(arm.lora_r)
    natural = load_benchmark(args.natural, args.natural_root)
    xm_dir = Path(args.xm_dir)
    xm_train = [it.to_item(args.media_root) for it in load_frozen(xm_dir / "train.jsonl")]
    guard_items = [it.to_item(args.media_root) for it in load_frozen(xm_dir / "guard.jsonl")]
    planted, report = load_planted(args.planted)
    guard_set = GuardSet(guard_items, planted, report["use_fallback_reference"], lambda it: it.to_item(args.media_root))
    samplings = samplings_from(experiment, arm)
    fill = model["missing_modality_fill"]
    dev = [it.to_item(args.media_root) for it in load_frozen(xm_dir / "dev.jsonl")]
    external = None
    if arm.critic == "gemini":
        from .critic import JudgeExecutor
        from .judges import make_judge
        from .judges.media import view_media

        media_dir = Path(args.out) / "judge_media"
        external = JudgeExecutor(make_judge(experiment["judges"]["gemini"]), lambda it, m: view_media(it, m, media_dir))
    train(arm, backend, natural, xm_train, guard_set, samplings, xm_dev_scorer(dev, samplings, fill), args.seed, args.steps,
          args.out, external, fill=fill)
    if arm.trained:
        backend.save_adapter(Path(args.out) / "adapter")


if __name__ == "__main__":
    main()
