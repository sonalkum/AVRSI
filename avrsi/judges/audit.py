"""Blind audit against the rubric of PLAN §7.2. Never used in training."""
from __future__ import annotations

import random
from typing import Sequence

from ..critic import parse_pair
from ..parse import Item, question_prompt
from .base import Judge, Media

AUDIT_PROMPT = """Two anonymous reasoning paths answer the same question about this clip. Compare them on:
- hallucinated claims: statements not supported by the clip;
- evidence: claims matching what the clip contains;
- attribution: each claim placed in the right segment, audio or visual.

Question: {question}

Reasoning A:
{first}

Reasoning B:
{second}

Reply with exactly one of: A, B, Tie"""


def audit_pair(judge: Judge, item: Item, ours: str, baseline: str, media: Sequence[Media], rng: random.Random) -> float | None:
    """Win for `ours`: 1, 0.5 on a tie, 0; None on a parse failure. A/B order is randomised."""
    swapped = rng.random() < 0.5
    first, second = (baseline, ours) if swapped else (ours, baseline)
    verdict = parse_pair(judge.complete(AUDIT_PROMPT.format(question=question_prompt(item), first=first, second=second), media))
    if verdict is None:
        return None
    if verdict == "Tie":
        return 0.5
    return float((verdict == "A") != swapped)
