"""Segment and sentence parsing, dedupe, format reward, answer extraction (PLAN §3.3)."""
from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass, field

ALL_MODALITIES = ("audio", "video")
SEGMENTS = ("audio", "visual", "fusion", "answer")
_TAG = re.compile(r"</?(?:audio|visual|fusion|answer)>")
_EXPECTED_TAGS = [t for s in SEGMENTS for t in (f"<{s}>", f"</{s}>")]
_TIMESTAMP = re.compile(r"\[\d+(?:\.\d+)?s\]")
_SPLIT = re.compile(r"(?<=[.!?])\s+")
JACCARD_DROP = 0.8
MAX_FUSION_SENTENCES = 2


@dataclass
class Item:
    item_id: str
    question: str
    answer: str  # "yes"/"no", free text, or the canonical option text
    options: tuple[str, ...] = ()  # in presentation order
    is_xm: bool = False
    silence: bool = False
    video: str | None = None
    audio: str | None = None
    modalities: tuple[str, ...] = ALL_MODALITIES  # what the sample contains (CMM can be unimodal)
    meta: dict = field(default_factory=dict)


@dataclass
class ParsedPath:
    raw: str
    valid: bool
    audio: list[str] = field(default_factory=list)  # deduped
    visual: list[str] = field(default_factory=list)  # deduped
    fusion: list[str] = field(default_factory=list)
    answer: str | None = None


def normalise_tokens(text: str) -> list[str]:
    kept = [c for c in text.lower() if c not in string.punctuation and not unicodedata.category(c).startswith("P")]
    return "".join(kept).split()


def split_sentences(text: str) -> list[str]:
    parts = _SPLIT.split(_TIMESTAMP.sub("", text).strip())
    return [p.strip() for p in parts if normalise_tokens(p)]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def dedupe_indices(sentences: list[str], question: str | None = None) -> list[int]:
    """Indices kept. A sentence is compared with every earlier one, dropped or not. question=None
    disables the question rule (XM items); callers pass question text without MCQ options."""
    tokens = [normalise_tokens(s) for s in sentences]
    question_tokens = normalise_tokens(question) if question else None
    kept = []
    for i, t in enumerate(tokens):
        echoes = question_tokens is not None and jaccard(t, question_tokens) >= JACCARD_DROP
        if not echoes and all(jaccard(t, tokens[j]) < JACCARD_DROP for j in range(i)):
            kept.append(i)
    return kept


def dedupe(sentences: list[str], question: str | None = None) -> list[str]:
    return [sentences[i] for i in dedupe_indices(sentences, question)]


def _segments(text: str) -> dict[str, str] | None:
    tags = list(_TAG.finditer(text))
    if [m.group() for m in tags] != _EXPECTED_TAGS:
        return None
    outside = [text[: tags[0].start()], text[tags[-1].end():]]
    outside += [text[tags[i].end(): tags[i + 1].start()] for i in range(1, 7, 2)]
    if any(s.strip() for s in outside):
        return None
    return {s: text[tags[2 * i].end(): tags[2 * i + 1].start()] for i, s in enumerate(SEGMENTS)}


def valid_format(text: str) -> bool:
    """ValidFormat, on raw text before dedupe. Non-empty means at least one sentence survives the splitter."""
    seg = _segments(text)
    return (
        seg is not None
        and bool(split_sentences(seg["audio"]))
        and bool(split_sentences(seg["visual"]))
        and len(split_sentences(seg["fusion"])) <= MAX_FUSION_SENTENCES
    )


def r_format(text: str) -> float:
    return float(valid_format(text))


def parse_path(text: str, item: Item) -> ParsedPath:
    if not valid_format(text):
        return ParsedPath(raw=text, valid=False)
    seg = _segments(text)
    question = None if item.is_xm else item.question
    return ParsedPath(
        raw=text,
        valid=True,
        audio=dedupe(split_sentences(seg["audio"]), question),
        visual=dedupe(split_sentences(seg["visual"]), question),
        fusion=dedupe(split_sentences(seg["fusion"]), question),
        answer=seg["answer"].strip(),
    )


def render_path(path: ParsedPath) -> str:
    """Deduped path as critics see it."""
    return (
        f"<audio>{' '.join(path.audio)}</audio>\n<visual>{' '.join(path.visual)}</visual>\n"
        f"<fusion>{' '.join(path.fusion)}</fusion>\n<answer>{path.answer}</answer>"
    )


def canonical_answer(raw: str, item: Item) -> str | None:
    """Option text for MCQ (letter or text); normalised string otherwise."""
    tokens = normalise_tokens(raw)
    if not tokens:
        return None
    if not item.options:
        return " ".join(tokens)
    for i, option in enumerate(item.options):
        letter, text = chr(ord("a") + i), normalise_tokens(option)
        if tokens in ([letter], text, [letter] + text):
            return option
    return None


def gold_answer(item: Item) -> str:
    return item.answer if item.options else " ".join(normalise_tokens(item.answer))


def fallback_extract(text: str, item: Item) -> str | None:
    """Documented evaluation-only extractor (PLAN §3.3), same for all systems. In order:
    1. the last <answer>...</answer> that canonicalises;
    2. Yes/No items: the last whole-word yes or no in the text outside tags;
    3. MCQ: "answer is/: X" letter, then the option text mentioned last, then a trailing bare letter;
    4. free text: the last non-empty line.
    Never used by training rewards. The guard uses it only for θ_0's condition-1 reference under the
    appendix 20% rule."""
    yes_no = not item.options and gold_answer(item) in ("yes", "no")
    for tagged in reversed(re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)):
        answer = canonical_answer(tagged, item)
        if answer is not None and (answer in ("yes", "no") or not yes_no):
            return answer
    body = _TAG.sub(" ", text)
    if item.options:
        letters = "".join(chr(ord("A") + i) for i in range(len(item.options)))
        hit = re.findall(rf"answer\s*(?:is|:)\s*\(?([{letters}])\b", body, re.I)
        if hit:
            return item.options[ord(hit[-1].upper()) - ord("A")]
        positions = [(body.lower().rfind(o.lower()), o) for o in item.options]
        best = max(positions)
        if best[0] >= 0:
            return best[1]
        hit = re.search(rf"(?:^|\s)\(?([{letters}])\)?[.:]?\s*$", body)
        return item.options[ord(hit.group(1)) - ord("A")] if hit else None
    if yes_no:
        hits = re.findall(r"\b(yes|no)\b", body, re.I)
        return hits[-1].lower() if hits else None
    lines = [ln for ln in body.splitlines() if normalise_tokens(ln)]
    return " ".join(normalise_tokens(lines[-1])) if lines else None


def eval_answer(text: str, item: Item) -> str | None:
    """Evaluation answer: strict when well-formed, otherwise the fallback extractor."""
    path = parse_path(text, item)
    return canonical_answer(path.answer, item) if path.valid else fallback_extract(text, item)


def question_prompt(item: Item) -> str:
    if not item.options:
        return item.question
    lines = [f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(item.options)]
    return "\n".join([item.question, *lines])
