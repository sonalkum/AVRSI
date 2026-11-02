"""XM templates, swap planning, twin_id, planted paths, splits, muxing (PLAN §5.2, §9, appendix)."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .parse import Item, dedupe_indices, parse_path, split_sentences, valid_format

F1_T = "Is there a sound of {c}?"
F2_T = "Can you see {c}?"
F3_T = "Does what you hear match what you see?"
F4_T = "Where is {c} present?"
F4_OPTIONS = ("audio", "video", "both", "neither")
ROLE_OF_OPTION = {"audio": "correct_audio", "video": "correct_visual", "both": "both", "neither": "neither"}
ALIGNED, AUDIO_SWAP, VIDEO_SWAP, SILENCE = "aligned", "audio_swap", "video_swap", "silence"
SWAPPED = (AUDIO_SWAP, VIDEO_SWAP)
DEFAULT_SHARES = {"F1": 0.35, "F2": 0.35, "F3": 0.15, "F4": 0.15}
PRESENCE_SHARES = {"F1": 0.5, "F2": 0.5}
SPLIT_SIZES = {"train": 20000, "guard": 1024, "dev": 512, "gate": 500, "test": 5000}
PLANTED_TYPES = ("false_audio", "false_visual", "modality_swapped", "control")
GATE_PER_TYPE, GUARD_PER_TYPE = 125, 64
MIN_VALID_BASE_RATE = 0.2

AUDIO_STYLES = ("The sound of {c} can be heard.", "There is a clear sound of {c}.", "You can hear {c} in the audio.")
VISUAL_STYLES = ("The video shows {c}.", "There is a clear view of {c}.", "You can see {c} on screen.")


@dataclass
class Clip:
    clip_id: str
    source_video: str
    label: str
    siglip_score: float = 1.0
    partner_scores: dict = field(default_factory=dict)


@dataclass
class XMItem:
    item_id: str
    split: str
    family: str
    kind: str
    twin_id: str
    question: str
    answer: str
    answer_class: str
    cls: str
    video_clip: str
    audio_clip: str | None  # None = silent track
    visual_label: str
    audio_label: str | None
    sources: tuple[str, ...]  # source videos, for disjointness and bootstrap clusters
    options: tuple[str, ...] = ()
    follows_A: str | None = None
    follows_V: str | None = None
    planted_carrier: bool = False

    def to_item(self, media_root: str = "") -> Item:
        path = str(Path(media_root) / f"{self.item_id}.mp4")
        return Item(self.item_id, self.question, self.answer, self.options, is_xm=True, silence=self.kind == SILENCE,
                    video=path, meta=asdict(self))


def load_clips(path: str | Path) -> list[Clip]:
    """CSV columns: clip_id, source_video, label, siglip_score, partner_scores (JSON)."""
    with open(path, newline="") as f:
        return [Clip(r["clip_id"], r["source_video"], r["label"], float(r["siglip_score"]),
                     json.loads(r.get("partner_scores") or "{}")) for r in csv.DictReader(f)]


def verified_clips(clips: Iterable[Clip], threshold: float, excluded_classes: Sequence[str] = ()) -> list[Clip]:
    """Keep clips whose class SigLIP verifies; classes under 0.9 precision are passed as excluded."""
    return [c for c in clips if c.siglip_score >= threshold and c.label not in excluded_classes]


def accept_class_precision(precision: float, minimum: float = 0.9) -> bool:
    return precision >= minimum


def label_disjoint_pairs(similarity: dict[frozenset, float], tier: float = 0.5) -> dict[str, set[str]]:
    """Label-disjoint tier: the bottom `tier` of class pairs by similarity (appendix)."""
    ranked = sorted(similarity, key=lambda p: (similarity[p], sorted(p)))
    partners: dict[str, set[str]] = {}
    for pair in ranked[: int(len(ranked) * tier)]:
        a, b = sorted(pair)
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def hash_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def overlap_filter(clips: Iterable[Clip], known_ids: Iterable[str]) -> tuple[list[Clip], list[Clip]]:
    """Phase 0b: drop clips whose clip or source id hash is seen in another set. Returns (kept, dropped)."""
    known = {hash_id(k) for k in known_ids}
    kept, dropped = [], []
    for clip in clips:
        hit = hash_id(clip.clip_id) in known or hash_id(clip.source_video) in known
        (dropped if hit else kept).append(clip)
    return kept, dropped


def split_sources(clips: Sequence[Clip], sizes: dict[str, int], seed: int,
                  exclude_sources: Iterable[str] = ()) -> dict[str, list[Clip]]:
    """Assign whole source videos to splits, in proportion to split size."""
    excluded = set(exclude_sources)
    by_source: dict[str, list[Clip]] = {}
    for clip in clips:
        if clip.source_video not in excluded:
            by_source.setdefault(clip.source_video, []).append(clip)
    sources = sorted(by_source)
    random.Random(seed).shuffle(sources)
    total, start, used = sum(sizes.values()), 0, 0
    result = {}
    for name, size in sizes.items():
        used += size
        stop = len(sources) * used // total
        result[name] = [c for s in sources[start:stop] for c in by_source[s]]
        start = stop
    return result


def assert_source_disjoint(splits: dict[str, Sequence[XMItem]], external_sources: Iterable[str] = ()) -> None:
    seen: dict[str, str] = {s: "external" for s in external_sources}
    for name, items in splits.items():
        for source in {s for it in items for s in it.sources}:
            if seen.setdefault(source, name) != name:
                raise ValueError(f"source {source} in both {seen[source]} and {name}")


def _largest_remainder(total: int, shares: dict[str, float]) -> dict[str, int]:
    exact = {k: total * v for k, v in shares.items()}
    counts = {k: int(v) for k, v in exact.items()}
    left = total - sum(counts.values())
    for k in sorted(exact, key=lambda k: counts[k] - exact[k])[:left]:
        counts[k] += 1
    return counts


def _pair_specs(n_pairs: int, shares: dict[str, float], silence_share: float, rng: random.Random) -> list[tuple]:
    """(family, kind, variant) per twin pair. Silence is `silence_share` of all F1 items, so twice
    that of F1 pairs. Answers are chosen so F1/F2 are 50/50 within conflict and aligned items alike."""
    counts = _largest_remainder(n_pairs, shares)
    specs = []
    n1 = counts.get("F1", 0)
    n_silence = round(2 * silence_share * n1)
    n_yes = min(n1 - n_silence, round(n1 / 2))
    specs += [("F1", SILENCE, "no")] * n_silence + [("F1", AUDIO_SWAP, "yes")] * n_yes
    specs += [("F1", AUDIO_SWAP, "no")] * (n1 - n_silence - n_yes)
    n2 = counts.get("F2", 0)
    specs += [("F2", VIDEO_SWAP, "yes")] * round(n2 / 2) + [("F2", VIDEO_SWAP, "no")] * (n2 - round(n2 / 2))
    specs += [("F3", SWAPPED[i % 2], "no") for i in range(counts.get("F3", 0))]
    f4_cycle = [(AUDIO_SWAP, "audio"), (AUDIO_SWAP, "visual"), (VIDEO_SWAP, "audio"), (VIDEO_SWAP, "visual")]
    specs += [("F4", *f4_cycle[i % 4]) for i in range(counts.get("F4", 0))]
    rng.shuffle(specs)
    return specs


def _yes_no(present: bool) -> str:
    return "yes" if present else "no"


def _f4_role(cls: str, heard: str | None, seen: str) -> str:
    in_audio, in_video = cls == heard, cls == seen
    return "both" if in_audio and in_video else "audio" if in_audio else "video" if in_video else "neither"


def _item(split: str, pair: int, tag: str, family: str, kind: str, cls: str, source: Clip, partner: Clip | None,
          options: tuple[str, ...]) -> XMItem:
    """Answers follow from the labels of (video, audio) alone, which the table below defines:
    kind       video   audio
    aligned    source  source
    audio_swap source  partner
    video_swap partner source
    silence    source  none"""
    video = partner if kind == VIDEO_SWAP else source
    audio = None if kind == SILENCE else partner if kind == AUDIO_SWAP else source
    seen, heard = video.label, audio.label if audio else None
    if family == "F1":
        question, answer = F1_T.format(c=cls), _yes_no(cls == heard)
        answer_class, by_a, by_v = answer, _yes_no(cls == heard), _yes_no(cls == seen)
    elif family == "F2":
        question, answer = F2_T.format(c=cls), _yes_no(cls == seen)
        answer_class, by_a, by_v = answer, _yes_no(cls == heard), _yes_no(cls == seen)
    elif family == "F3":
        question, answer = F3_T, _yes_no(heard == seen)
        answer_class, by_a, by_v = answer, None, None
    else:
        question, answer = F4_T.format(c=cls), _f4_role(cls, heard, seen)
        answer_class = ROLE_OF_OPTION[answer]
        by_a, by_v = ("audio" if cls == heard else "neither"), ("video" if cls == seen else "neither")
    sources = tuple(dict.fromkeys(s.source_video for s in (source, partner) if s is not None and (s is video or s is audio)))
    return XMItem(f"{split}-{pair:06d}-{tag}", split, family, kind, f"{split}-{pair:06d}", question, answer, answer_class,
                  cls, video.clip_id, audio.clip_id if audio else None, seen, heard, sources, options, by_a, by_v)


def _question_class(family: str, kind: str, variant: str, source: Clip, partner: Clip) -> str:
    """Class asked about so the swapped item gets the intended answer."""
    if kind == SILENCE:
        return source.label
    if family == "F4":
        heard = partner.label if kind == AUDIO_SWAP else source.label
        seen = source.label if kind == AUDIO_SWAP else partner.label
        return heard if variant == "audio" else seen
    if family == "F3":
        return source.label
    # F1 audio_swap and F2 video_swap: the partner's class is the one present after the swap.
    return partner.label if variant == "yes" else source.label


def _visually_absent(clip: Clip, cls: str, threshold: float) -> bool:
    """§5.2: a visual "No" needs a low SigLIP score on the partner class."""
    return clip.partner_scores.get(cls, 0.0) < threshold


def build_split(clips: Sequence[Clip], split: str, n_items: int, disjoint: dict[str, set[str]], rng: random.Random,
                shares: dict[str, float] | None = None, silence_share: float = 0.2, threshold: float = 1.0,
                first_pair: int = 0, planted_carrier: bool = False) -> list[XMItem]:
    """n_items/2 twin pairs: a swapped or silence item plus its aligned twin, same question."""
    by_class: dict[str, list[Clip]] = {}
    for clip in clips:
        by_class.setdefault(clip.label, []).append(clip)
    pool = sorted(clips, key=lambda c: c.clip_id)
    rng.shuffle(pool)
    items: list[XMItem] = []
    specs = _pair_specs(n_items // 2, shares or DEFAULT_SHARES, silence_share, rng)
    for i, (family, kind, variant) in enumerate(specs):
        source, partner = _draw_pair(pool, i, by_class, disjoint, rng, threshold)
        cls = _question_class(family, kind, variant, source, partner)
        options = tuple(rng.sample(F4_OPTIONS, 4)) if family == "F4" else ()
        pair = first_pair + i
        items.append(_item(split, pair, "x", family, kind, cls, source, partner, options))
        items.append(_item(split, pair, "a", family, ALIGNED, cls, source, partner, options))
    if planted_carrier:
        for item in items:
            item.planted_carrier = True
    return items


def _draw_pair(pool, i, by_class, disjoint, rng, threshold) -> tuple[Clip, Clip]:
    for offset in range(len(pool)):
        source = pool[(i + offset) % len(pool)]
        classes = sorted(c for c in disjoint.get(source.label, ()) if by_class.get(c))
        rng.shuffle(classes)
        for cls in classes:
            for partner in rng.sample(by_class[cls], len(by_class[cls])):
                if partner.source_video != source.source_video and _visually_absent(source, cls, threshold) \
                        and _visually_absent(partner, source.label, threshold):
                    return source, partner
    raise ValueError("no label-disjoint partner available in this split")


def build_xm(splits: dict[str, Sequence[Clip]], disjoint: dict[str, set[str]], seed: int, threshold: float = 1.0,
             sizes: dict[str, int] | None = None, silence_share: float = 0.2,
             guard_planted: int = 256) -> dict[str, list[XMItem]]:
    """All XM splits. XM-Gate and XM-Guard's planted items use F1/F2 without silence (appendix)."""
    sizes = sizes or SPLIT_SIZES
    rng = random.Random(seed)
    result = {}
    for name in sizes:
        common = dict(disjoint=disjoint, rng=rng, threshold=threshold)
        if name == "gate":
            result[name] = build_split(splits[name], name, sizes[name], shares=PRESENCE_SHARES, silence_share=0, **common)
        elif name == "guard":
            carriers = build_split(splits[name], name, guard_planted, shares=PRESENCE_SHARES, silence_share=0,
                                   planted_carrier=True, **common)
            rest = build_split(splits[name], name, sizes[name] - guard_planted, silence_share=silence_share,
                               first_pair=guard_planted // 2, **common)
            result[name] = carriers + rest
        else:
            result[name] = build_split(splits[name], name, sizes[name], silence_share=silence_share, **common)
    return result


def sample_batch(natural: Sequence[Item], xm_train: Sequence[Item], B: int, p: float, rng: random.Random) -> list[Item]:
    """§3.6: each slot is XM-Train with Bernoulli probability p, else natural."""
    return [rng.choice(xm_train if xm_train and rng.random() < p else natural) for _ in range(B)]


@dataclass
class PlantedPath:
    item: XMItem
    ptype: str
    segment: str  # "audio" or "visual"
    base_text: str
    planted_text: str
    sentence: str
    index: int  # position of the planted sentence among the segment's deduped sentences


def _tagged(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return match.group(1).strip() if match else None


def assemble_base(text: str) -> str | None:
    """Appendix 20% rule: base from one exemplar rollout's own sentences. Fusion keeps its first two."""
    audio, visual = _tagged(text, "audio"), _tagged(text, "visual")
    if not audio or not visual:
        return None
    fusion = " ".join(split_sentences(_tagged(text, "fusion") or "")[:2])
    assembled = f"<audio>{audio}</audio>\n<visual>{visual}</visual>\n<fusion>{fusion}</fusion>\n<answer>{_tagged(text, 'answer') or ''}</answer>"
    return assembled if valid_format(assembled) else None


def select_base_paths(theta0: dict[str, Sequence[str]], exemplar: dict[str, str] | None = None) -> tuple[dict[str, str], dict]:
    """theta0 maps an item to its θ_0 rollouts, several draws each since every carrier needs a base.
    Takes each item's first valid draw, unless under 20% of draws pass ValidFormat (appendix)."""
    draws = [t for texts in theta0.values() for t in texts]
    rate = sum(map(valid_format, draws)) / len(draws) if draws else 0.0
    if rate >= MIN_VALID_BASE_RATE:
        valid = {k: next(t for t in texts if valid_format(t)) for k, texts in theta0.items() if any(map(valid_format, texts))}
        return valid, {"valid_rate": rate, "mode": "theta0", "use_fallback_reference": False}
    if exemplar is None:
        raise ValueError(f"only {rate:.1%} of θ_0 paths pass ValidFormat; exemplar rollouts required")
    assembled = {k: b for k, v in exemplar.items() if (b := assemble_base(v)) is not None}
    return assembled, {"valid_rate": rate, "mode": "exemplar", "use_fallback_reference": True}


def _insert(base_text: str, segment: str, sentence: str, position: int | None, rng: random.Random) -> tuple[str, int]:
    match = re.search(rf"<{segment}>(.*?)</{segment}>", base_text, re.S)
    sentences = split_sentences(match.group(1))
    at = rng.randint(0, len(sentences)) if position is None else position
    sentences.insert(at, sentence)
    fixed = [s if s[-1] in ".!?" else s + "." for s in sentences]
    return base_text[: match.start(1)] + " ".join(fixed) + base_text[match.end(1):], at


def _plant(item: XMItem, base_text: str, ptype: str, segment: str, classes: Sequence[str], rng: random.Random,
           attempts: int = 50) -> PlantedPath:
    styles = AUDIO_STYLES if segment == "audio" else VISUAL_STYLES
    probe = Item(item.item_id, item.question, item.answer, is_xm=True)
    for _ in range(attempts):
        sentence = rng.choice(styles).format(c=rng.choice(list(classes)))
        text, at = _insert(base_text, segment, sentence, None, rng)
        raw = split_sentences(_tagged(text, segment))
        kept = dedupe_indices(raw)  # XM items have no question rule
        if at in kept:
            assert parse_path(text, probe).valid
            return PlantedPath(item, ptype, segment, base_text, text, sentence, kept.index(at))
    raise ValueError(f"{item.item_id}: planted sentence kept being deduped")


def _false_classes(item: XMItem, disjoint: dict[str, set[str]]) -> list[str]:
    labels = [l for l in (item.audio_label, item.visual_label) if l]
    options = set.intersection(*(disjoint.get(l, set()) for l in labels)) - set(labels)
    if not options:
        raise ValueError(f"{item.item_id}: no label-disjoint false class")
    return sorted(options)


def build_planted(items: Sequence[XMItem], base_paths: dict[str, str], disjoint: dict[str, set[str]], rng: random.Random,
                  per_type: int) -> list[PlantedPath]:
    """§9: four types, per_type each. Swapped-type items sit on swapped clips; silence clips are
    excluded (appendix); questions are F1/F2 only."""
    usable = [it for it in items if it.family in ("F1", "F2") and it.kind != SILENCE and it.item_id in base_paths]
    rng.shuffle(usable)
    swapped = [it for it in usable if it.kind in SWAPPED][:per_type]
    if len(swapped) < per_type:
        raise ValueError(f"need {per_type} swapped carriers, have {len(swapped)}")
    taken = {it.item_id for it in swapped}
    rest = [it for it in usable if it.item_id not in taken]
    if len(rest) < 3 * per_type:
        raise ValueError(f"need {3 * per_type} carriers, have {len(rest)}")
    planted = []
    half = (per_type + 1) // 2  # 63/62 and 32/32
    for i, item in enumerate(swapped):
        segment = "audio" if i < half else "visual"
        # Visible but not audible in <audio>; audible but not visible in <visual>.
        cls = item.visual_label if segment == "audio" else item.audio_label
        planted.append(_plant(item, base_paths[item.item_id], "modality_swapped", segment, [cls], rng))
    for t, ptype in enumerate(("false_audio", "false_visual", "control")):
        for i, item in enumerate(rest[t * per_type: (t + 1) * per_type]):
            if ptype == "control":
                segment = "audio" if i < half else "visual"
                classes = [item.audio_label if segment == "audio" else item.visual_label]
            else:
                segment = "audio" if ptype == "false_audio" else "visual"
                classes = _false_classes(item, disjoint)
            planted.append(_plant(item, base_paths[item.item_id], ptype, segment, classes, rng))
    return planted


def mux_command(video_src: str, audio_src: str | None, out: str) -> list[str]:
    """Video stream of video_src with the audio of audio_src, or silence over the whole track."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_src]
    cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"] if audio_src is None else ["-i", audio_src]
    return cmd + ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", out]


def strip_audio_command(video_src: str, out: str) -> list[str]:
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", video_src, "-map", "0:v:0", "-c:v", "copy", "-an", out]


def extract_audio_command(video_src: str, out: str) -> list[str]:
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", video_src, "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le", out]


def mux_item(item: XMItem, clip_path: Callable[[str], str], out_dir: str | Path,
             runner: Callable = subprocess.run) -> Path:
    out = Path(out_dir) / f"{item.item_id}.mp4"
    audio = clip_path(item.audio_clip) if item.audio_clip else None
    runner(mux_command(clip_path(item.video_clip), audio, str(out)), check=True)
    return out


def freeze(items: Sequence[XMItem], path: str | Path) -> str:
    """Write a split as canonical JSONL plus its SHA-256 (Phase 0b step 8). Refuses to overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} is frozen")
    lines = [json.dumps(asdict(it), sort_keys=True) for it in sorted(items, key=lambda it: it.item_id)]
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    Path(f"{path}.sha256").write_text(digest + "\n")
    return digest


def load_frozen(path: str | Path) -> list[XMItem]:
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != Path(f"{path}.sha256").read_text().strip():
        raise ValueError(f"{path} does not match its frozen hash")
    rows = [json.loads(line) for line in data.decode().splitlines()]
    return [XMItem(**{**r, "sources": tuple(r["sources"]), "options": tuple(r["options"])}) for r in rows]


def human_check_passes(errors: int, checked: int = 500, max_error: float = 0.05) -> bool:
    return errors / checked <= max_error


def save_planted(planted: Sequence[PlantedPath], report: dict, path: str | Path) -> None:
    rows = [{**asdict(p), "item": asdict(p.item)} for p in planted]
    Path(path).write_text(json.dumps({"report": report, "planted": rows}))


def load_planted(path: str | Path) -> tuple[list[PlantedPath], dict]:
    data = json.loads(Path(path).read_text())
    planted = []
    for row in data["planted"]:
        item = XMItem(**{**row["item"], "sources": tuple(row["item"]["sources"]), "options": tuple(row["item"]["options"])})
        planted.append(PlantedPath(**{**row, "item": item}))
    return planted, data["report"]
