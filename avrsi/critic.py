"""Frozen prompts, verdict parsing, critic views, pass planner (PLAN §3.2, §3.4)."""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .models.base import CRITIC, Backend, Generation, Sampling, build_model_input
from .parse import ALL_MODALITIES, Item, ParsedPath, question_prompt, render_path

VERDICTS = ("supported", "unsupported", "unverifiable", "vacuous")
A_CHECK, V_CHECK, J_CHECK, AV_PAIR = "a_check", "v_check", "j_check", "av_pair"
VIEW_MODALITIES = {
    A_CHECK: ("audio",),
    V_CHECK: ("video",),
    J_CHECK: ALL_MODALITIES,
    AV_PAIR: ALL_MODALITIES,
}
GUARD_ACTOR_GENERATIONS, GUARD_CHECK_PASSES, GUARD_PAIR_PASSES = 1024, 256, 512

ACTOR_PROMPT = """Answer the question about this clip. Reply in exactly this format, with nothing outside the tags:
<audio>sentences citing only what is heard</audio>
<visual>sentences citing only what is seen</visual>
<fusion>at most 2 sentences on how the evidence combines, naming any conflict</fusion>
<answer>your final answer</answer>

Question: {question}"""

ACTOR_EXEMPLAR = """Example of the format:
<audio>A dog barks twice. A car passes in the distance.</audio>
<visual>A man walks a dog along a road.</visual>
<fusion>The barking fits the dog on screen.</fusion>
<answer>Yes</answer>

"""  # appendix: planted-base generation only, under the 20% rule

CHECK_PROMPT = """You are checking claims about a clip against what is {sense}.

Question asked about the clip: {question}

Claims:
{claims}

Give one verdict per claim:
supported: {sense} in the clip and it confirms the claim.
unsupported: {sense} in the clip and it contradicts the claim, or the claimed thing is absent.
unverifiable: cannot be judged from what is {sense}.
vacuous: true of any clip, says nothing specific.

Reply with one line per claim, nothing else, as: <number>: <verdict>"""

PAIR_PROMPT = """Two reasoning paths answer the same question about this clip. Judge which reasoning is better: accurate to what is heard and seen, informative, and logically sound.

Question: {question}

Reasoning A:
{first}

Reasoning B:
{second}

Reply with exactly one of: A, B, Tie"""

_SENSE = {"audio": "heard", "visual": "seen"}


def prompt_digest() -> str:
    return hashlib.sha256("\0".join([ACTOR_PROMPT, ACTOR_EXEMPLAR, CHECK_PROMPT, PAIR_PROMPT]).encode()).hexdigest()


def actor_prompt(item: Item, exemplar: bool = False) -> str:
    text = ACTOR_PROMPT.format(question=question_prompt(item))
    return ACTOR_EXEMPLAR + text if exemplar else text


def check_prompt(item: Item, segment: str, sentences: Sequence[str]) -> str:
    """Shared by A-, V- and J-check: a J-check of a segment gets the same text as the isolated check."""
    claims = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return CHECK_PROMPT.format(sense=_SENSE[segment], question=question_prompt(item), claims=claims)


def pair_prompt(item: Item, first: ParsedPath, second: ParsedPath) -> str:
    return PAIR_PROMPT.format(question=question_prompt(item), first=render_path(first), second=render_path(second))


_CHECK_LINE = re.compile(r"^\s*(\d+)\s*[:.)]\s*([A-Za-z]+)\s*\.?\s*$")


def parse_check(text: str, n: int) -> list[str] | None:
    """Verdict per sentence, or None on parse failure or sentence-count mismatch."""
    numbered = [m for m in map(_CHECK_LINE.match, text.splitlines()) if m]
    verdicts = [m.group(2).lower() for m in numbered]
    if [int(m.group(1)) for m in numbered] != list(range(1, n + 1)) or any(v not in VERDICTS for v in verdicts):
        return None
    return verdicts


def verdicts_for_reward(parsed: list[str] | None, n: int) -> list[str]:
    """§3.3: in the reward, a failure makes the whole segment unverifiable. Gate 0 and the guard
    exclude failures instead and must not call this."""
    return ["unverifiable"] * n if parsed is None else parsed


def parse_pair(text: str) -> str | None:
    word = text.strip().rstrip(".!").strip().lower()
    return {"a": "A", "b": "B", "tie": "Tie"}.get(word)


@dataclass(frozen=True)
class CriticPass:
    view: str
    prompt: str
    path: int | None = None
    segment: str | None = None
    n_sentences: int = 0
    pair: tuple[int, int] | None = None  # (shown as A, shown as B)

    @property
    def modalities(self) -> tuple[str, ...]:
        return VIEW_MODALITIES[self.view]


def check_view_for(segment: str, mode: str) -> str:
    if mode == "joint":
        return J_CHECK
    return A_CHECK if segment == "audio" else V_CHECK


def plan_passes(arm, paths: Sequence[ParsedPath], item: Item, rng: random.Random) -> list[CriticPass]:
    """Exact passes for one question. Malformed paths and empty segments get none."""
    passes = []
    if arm.pair_orders != "none":
        for j in range(len(paths)):
            for k in range(j + 1, len(paths)):
                if not (paths[j].valid and paths[k].valid):
                    continue
                if arm.pair_orders == "both":
                    orders = [(j, k), (k, j)]
                else:
                    orders = [(j, k) if rng.random() < 0.5 else (k, j)]
                for a, b in orders:
                    passes.append(CriticPass(AV_PAIR, pair_prompt(item, paths[a], paths[b]), pair=(a, b)))
    if arm.check_view != "none":
        for i, path in enumerate(paths):
            if not path.valid:
                continue
            for segment in ("audio", "visual"):
                sentences = getattr(path, segment)
                if sentences:
                    view = check_view_for(segment, arm.check_view)
                    prompt = check_prompt(item, segment, sentences)
                    passes.append(CriticPass(view, prompt, path=i, segment=segment, n_sentences=len(sentences)))
    return passes


def max_passes(arm) -> int:
    path = ParsedPath(raw="", valid=True, audio=["a claim."], visual=["a claim."], fusion=[], answer="yes")
    return len(plan_passes(arm, [path] * arm.K, Item("budget", "q", "yes"), random.Random(0)))


def run_volume(arm, steps: int | None = None) -> int:
    return max_passes(arm) * arm.B * (arm.T_max if steps is None else steps)


def guard_check_volume() -> int:
    return GUARD_ACTOR_GENERATIONS + GUARD_CHECK_PASSES + GUARD_PAIR_PASSES


def guard_run_volume(T: int, T_max: int) -> int:
    """Checks at every multiple of T plus θ_0's reference."""
    return (T_max // T + 1) * guard_check_volume()


class Executor(Protocol):
    def run(self, item: Item, passes: Sequence[CriticPass]) -> list[str]: ...


class BackendExecutor:
    """Runs passes on a Backend, deterministic greedy. role=CRITIC for rewards, ACTOR when the guard
    tests the actor as a judge."""

    def __init__(self, backend: Backend, sampling: Sampling, role: str = CRITIC, fill: str | None = None):
        self.backend, self.sampling, self.role, self.fill = backend, sampling, role, fill
        self.calls = 0

    def run(self, item: Item, passes: Sequence[CriticPass]) -> list[str]:
        inputs = [build_model_input(item, p.modalities, p.prompt, self.fill) for p in passes]
        self.calls += len(inputs)
        return [g.text for g in self.backend.generate(inputs, self.sampling, self.role)] if inputs else []


class JudgeExecutor:
    """External critic for A3/A4. media_for(item, modalities) must return only the allowed streams;
    judges.media.view_media does that for files on disk."""

    def __init__(self, judge, media_for):
        self.judge, self.media_for = judge, media_for
        self.calls = 0

    def run(self, item: Item, passes: Sequence[CriticPass]) -> list[str]:
        self.calls += len(passes)
        return [self.judge.complete(p.prompt, self.media_for(item, p.modalities)) for p in passes]


@dataclass
class GroupVerdicts:
    audio: list[list[str]]
    visual: list[list[str]]
    pair_outcomes: dict[tuple[int, int], str | None] = field(default_factory=dict)
    passes: int = 0
    check_failures: int = 0
    pair_failures: int = 0


def score_group(arm, executor: Executor, item: Item, paths: Sequence[ParsedPath], rng: random.Random) -> GroupVerdicts:
    passes = plan_passes(arm, paths, item, rng)
    outputs = executor.run(item, passes)
    result = GroupVerdicts(audio=[[] for _ in paths], visual=[[] for _ in paths], passes=len(passes))
    for i, path in enumerate(paths):  # segments never checked stay fully unverified
        result.audio[i] = ["unverifiable"] * len(path.audio)
        result.visual[i] = ["unverifiable"] * len(path.visual)
    for p, text in zip(passes, outputs):
        if p.view == AV_PAIR:
            verdict = parse_pair(text)
            result.pair_outcomes[p.pair] = verdict
            result.pair_failures += verdict is None
        else:
            parsed = parse_check(text, p.n_sentences)
            result.check_failures += parsed is None
            target = result.audio if p.segment == "audio" else result.visual
            target[p.path] = verdicts_for_reward(parsed, p.n_sentences)
    return result


def unsupported_score(
    backend: Backend, item: Item, check: CriticPass, index: int, verdicts: list[str] | None,
    sampled_outputs: Sequence[str] | None, role: str = CRITIC, fill: str | None = None,
) -> float:
    """§9 AUROC score: log-probability of the "unsupported" tokens at the planted sentence, or, if
    unavailable, the fraction of "unsupported" across the sampled passes."""
    if backend.supports_logprob_scoring and verdicts is not None:
        prefix = "".join(f"{i + 1}: {verdicts[i]}\n" for i in range(index)) + f"{index + 1}:"
        completion = prefix + " unsupported"
        ids, n_verdict = backend.encode(completion), len(backend.encode(" unsupported"))
        inputs = [build_model_input(item, check.modalities, check.prompt, fill)]
        logps = backend.score_logprobs(inputs, [Generation(completion, ids)], role)[0]
        return float(logps[-n_verdict:].sum())
    if sampled_outputs is None:
        raise ValueError("no log-probabilities and no sampled passes")
    parsed = [parse_check(t, check.n_sentences) for t in sampled_outputs]
    usable = [p for p in parsed if p is not None]
    return sum(p[index] == "unsupported" for p in usable) / len(usable) if usable else float("nan")


