"""Periodic sync and guard (PLAN §3.5, §8). The guard scores the actor θ, as answerer and as judge."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from .critic import (AV_PAIR, BackendExecutor, CriticPass, actor_prompt, check_prompt, check_view_for, pair_prompt, parse_check,
                     parse_pair, unsupported_score)
from .metrics import PlantedResult, order_consistency, planted_balanced_accuracy, planted_summary
from .models.base import ACTOR, REFERENCE, Backend, Sampling, build_model_input
from .parse import Item, dedupe, eval_answer, gold_answer, parse_path
from .perturb import PlantedPath, XMItem
from .rewards import r_outcome, s_m

MARGIN = 0.02
DRIFT_MARGIN = 0.05
BLOCK_ALARM = 0.25
_EPS = 1e-12
VACUOUS_PROBE = "Something is happening in this clip."


@dataclass
class GuardSet:
    items: list[Item]  # XM-Guard
    planted: list[PlantedPath]
    use_fallback_reference: bool = False  # appendix 20% rule, θ_0 condition 1 only
    resolve: Callable[[XMItem], Item] = XMItem.to_item


@dataclass
class GuardScores:
    outcome: float
    planted_ba: float | None = None
    order_consistency: float | None = None
    parse_failure_rate: float | None = None
    generations: int = 0
    check_passes: int = 0
    pair_passes: int = 0


@dataclass
class Decision:
    step: int
    handoff: bool
    condition_1: bool
    condition_2: bool | None
    actor: dict
    incumbent: dict
    drift_alarm: bool = False
    block_rate: float = 0.0
    block_alarm: bool = False


def planted_results(executor, planted: Sequence[PlantedPath], mode: str, resolve=XMItem.to_item) -> list[PlantedResult]:
    """One pass per planted path, on the planted segment, in the arm's own checking view."""
    results = []
    for p in planted:
        item = resolve(p.item)
        sentences = getattr(parse_path(p.planted_text, item), p.segment)
        prompt = check_prompt(item, p.segment, sentences)
        check = CriticPass(check_view_for(p.segment, mode), prompt, segment=p.segment, n_sentences=len(sentences))
        parsed = parse_check(executor.run(item, [check])[0], len(sentences))
        results.append(PlantedResult(p.ptype, p.segment, parsed[p.index] if parsed else None))
    return results


def pair_order_consistency(executor, planted: Sequence[PlantedPath], resolve=XMItem.to_item) -> tuple[float, dict]:
    """Planted path vs its original, both orders: 2 passes per path."""
    forward, backward, by_segment = [], [], {"audio": ([], []), "visual": ([], [])}
    for p in planted:
        item = resolve(p.item)
        path, original = parse_path(p.planted_text, item), parse_path(p.base_text, item)
        passes = [CriticPass(AV_PAIR, pair_prompt(item, path, original), pair=(0, 1)),
                  CriticPass(AV_PAIR, pair_prompt(item, original, path), pair=(1, 0))]
        first, second = map(parse_pair, executor.run(item, passes))
        forward.append(first), backward.append(second)
        by_segment[p.segment][0].append(first), by_segment[p.segment][1].append(second)
    per_segment = {s: order_consistency(*v) for s, v in by_segment.items() if v[0]}
    return order_consistency(forward, backward), per_segment


def actor_outcome(backend: Backend, items: Sequence[Item], prompt_for, sampling: Sampling, fill: str | None = None,
                  use_fallback: bool = False, role: str = ACTOR) -> tuple[float, int]:
    """Strict outcome accuracy. The fallback runs only for θ_0's reference under the 20% rule."""
    inputs = [build_model_input(item, item.modalities, prompt_for(item), fill) for item in items]
    outputs = backend.generate(inputs, sampling, role)
    correct = 0
    for item, out in zip(items, outputs):
        if use_fallback:
            correct += eval_answer(out.text, item) == gold_answer(item)
        else:
            correct += r_outcome(parse_path(out.text, item), item)
    return correct / len(items), len(items)


def evaluate_actor(backend: Backend, guard_set: GuardSet, arm, prompt_for, greedy: Sampling, critic_greedy: Sampling,
                   fill: str | None = None, reference: bool = False) -> GuardScores:
    """Full XM-Guard check under θ: 1,024 generations, 256 check passes, 512 AV-pair passes."""
    fallback = reference and guard_set.use_fallback_reference
    outcome, generations = actor_outcome(backend, guard_set.items, prompt_for, greedy, fill, fallback)
    judge = BackendExecutor(backend, critic_greedy, role=ACTOR, fill=fill)
    results = planted_results(judge, guard_set.planted, arm.guard_view, guard_set.resolve)
    checks = judge.calls
    consistency, _ = pair_order_consistency(judge, guard_set.planted, guard_set.resolve)
    failures = [planted_summary(results, s)["parse_failure_rate"] for s in ("audio", "visual")]
    return GuardScores(outcome, planted_balanced_accuracy(results), consistency, sum(failures) / 2, generations, checks,
                       judge.calls - checks)


class Guard:
    def __init__(self, arm, log_path: str | Path | None = None):
        self.arm = arm
        self.log_path = Path(log_path) if log_path else None
        self.theta0: GuardScores | None = None
        self.incumbent: GuardScores | None = None
        self.decisions: list[Decision] = []

    def _log(self, record: dict) -> None:
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    def set_reference(self, scores: GuardScores) -> None:
        self.theta0 = self.incumbent = scores
        self._log({"event": "theta0_reference", **asdict(scores)})

    def _record(self, decision: Decision) -> Decision:
        self.decisions.append(decision)
        decision.block_rate = sum(not d.handoff for d in self.decisions) / len(self.decisions)
        decision.block_alarm = decision.block_rate > BLOCK_ALARM
        self._log({"event": "decision", **asdict(decision)})
        return decision

    def decide(self, step: int, scores: GuardScores) -> Decision:
        inc, ref = self.incumbent, self.theta0
        cond1 = scores.outcome >= inc.outcome - MARGIN - _EPS
        cond2 = scores.planted_ba >= inc.planted_ba - MARGIN - _EPS
        drift = scores.outcome < ref.outcome - DRIFT_MARGIN - _EPS or scores.planted_ba < ref.planted_ba - DRIFT_MARGIN - _EPS
        decision = Decision(step, cond1 and cond2, cond1, cond2, asdict(scores), asdict(inc), drift_alarm=drift)
        if decision.handoff:
            self.incumbent = scores  # incumbent scores are reused from its own check
        return self._record(decision)

    def decide_minibatch(self, step: int, actor_accuracy: float, critic_accuracy: float) -> Decision:
        """A10: AudioRSI's non-strict outcome test on one mini-batch; no condition 2, no θ_0 reference."""
        passed = actor_accuracy >= critic_accuracy - _EPS
        return self._record(Decision(step, passed, passed, None, {"outcome": actor_accuracy}, {"outcome": critic_accuracy}))

    @property
    def block_rate(self) -> float:
        return sum(not d.handoff for d in self.decisions) / len(self.decisions) if self.decisions else 0.0


def rollout_draws(backend: Backend, items: Sequence[Item], sampling: Sampling, draws: int, fill: str | None = None,
                  exemplar: bool = False) -> dict[str, list[str]]:
    """θ_0 paths at rollout settings for planted bases (§9), several draws per item."""
    result = {}
    for item in items:
        model_input = build_model_input(item, item.modalities, actor_prompt(item, exemplar), fill)
        result[item.item_id] = [g.text for g in backend.generate([model_input] * draws, sampling, REFERENCE)]
    return result


def gate0_run(executor, pair_executor, planted: Sequence[PlantedPath], resolve=XMItem.to_item) -> dict:
    """§9 on one critic: A/V-check and J-check on the same planted items, plus AV-pair consistency.
    Returns gate0_report's inputs, without AUROC scores (see attach_auroc)."""
    isolated = planted_results(executor, planted, "isolated", resolve)
    joint = planted_results(executor, planted, "joint", resolve)
    before = getattr(pair_executor, "calls", 0)
    outputs, by_segment = [], {"audio": ([], []), "visual": ([], [])}
    for p in planted:
        item = resolve(p.item)
        path, original = parse_path(p.planted_text, item), parse_path(p.base_text, item)
        passes = [CriticPass(AV_PAIR, pair_prompt(item, path, original), pair=(0, 1)),
                  CriticPass(AV_PAIR, pair_prompt(item, original, path), pair=(1, 0))]
        first, second = pair_executor.run(item, passes)
        outputs += [first, second]
        by_segment[p.segment][0].append(parse_pair(first)), by_segment[p.segment][1].append(parse_pair(second))
    failures = sum(parse_pair(t) is None for t in outputs) / len(outputs) if outputs else float("nan")
    consistency = {s: order_consistency(*v) for s, v in by_segment.items() if v[0]}
    return {"isolated": isolated, "joint": joint, "order_consistency": consistency, "pair_parse_failure_rate": failures,
            "pair_passes": getattr(pair_executor, "calls", 0) - before}


def attach_auroc(backend: Backend, greedy: Sampling, sampled: Sampling, results: list[PlantedResult], planted: Sequence[PlantedPath], mode: str,
                 role: str, fill: str | None = None, resolve=XMItem.to_item, passes: int = 4) -> None:
    """§9 AUROC score: fills PlantedResult.score. Sampled passes run only without log-probabilities."""
    for result, p in zip(results, planted):
        item = resolve(p.item)
        sentences = getattr(parse_path(p.planted_text, item), p.segment)
        check = CriticPass(check_view_for(p.segment, mode), check_prompt(item, p.segment, sentences), segment=p.segment,
                           n_sentences=len(sentences))
        sampled_outputs = None
        verdicts = None
        if backend.supports_logprob_scoring and result.verdict is not None:
            output = BackendExecutor(backend, greedy, role=role, fill=fill).run(item, [check])[0]
            verdicts = parse_check(output, len(sentences))
        if verdicts is None:
            inputs = [build_model_input(item, check.modalities, check.prompt, fill)] * passes
            sampled_outputs = [g.text for g in backend.generate(inputs, sampled, role)]
        result.score = unsupported_score(backend, item, check, p.index, verdicts, sampled_outputs, role, fill)


def run_probes(executor, planted: Sequence[PlantedPath], resolve=XMItem.to_item) -> dict:
    """§9 probes on the XM-Gate control paths. Each answers its question as a rate."""
    controls = [p for p in planted if p.ptype == "control"]
    vacuous, laundered, leaked, raised = [], [], [], []
    for p in controls:
        item = resolve(p.item)
        path = parse_path(p.planted_text, item)
        base = getattr(path, p.segment)

        def verdicts(sentences):
            check = CriticPass(check_view_for(p.segment, "isolated"), check_prompt(item, p.segment, sentences),
                               segment=p.segment, n_sentences=len(sentences))
            return parse_check(executor.run(item, [check])[0], len(sentences))

        result = verdicts(base + [VACUOUS_PROBE])
        if result:
            vacuous.append(result[-1] == "vacuous")
        moved = replace(path, fusion=[f"The clip also contains {p.item.cls}."])
        pair = CriticPass(AV_PAIR, pair_prompt(item, moved, path), pair=(0, 1))
        verdict = parse_pair(executor.run(item, [pair])[0])
        if verdict is not None:
            laundered.append(verdict == "A")
        result = verdicts(base + [p.item.question.rstrip("?") + "."])
        if result:
            leaked.append(result[-1] == "supported")
        paraphrase = "Indeed, " + p.sentence[0].lower() + p.sentence[1:]
        original = verdicts(base)
        extended = verdicts(base + [paraphrase]) if len(dedupe(base + [paraphrase])) == len(base) + 1 else None
        if original and extended:
            raised.append(s_m(extended) > s_m(original))
    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    return {"vacuous_returned": mean(vacuous), "fusion_laundering_rewarded": mean(laundered),
            "question_leak_passed": mean(leaked), "paraphrase_raised_s_m": mean(raised)}
