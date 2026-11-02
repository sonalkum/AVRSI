import json
import random
from collections import Counter, defaultdict

import pytest
from conftest import GOOD, make_clips, make_disjoint

from avrsi.parse import Item, parse_path, valid_format
from avrsi.perturb import (ALIGNED, AUDIO_SWAP, F3_T, F4_OPTIONS, SILENCE, VIDEO_SWAP, Clip, assemble_base, assert_source_disjoint, build_planted,
                           build_split, extract_audio_command, freeze, human_check_passes, label_disjoint_pairs, load_clips, load_frozen,
                           load_planted, mux_command, mux_item, overlap_filter, sample_batch, save_planted, select_base_paths, split_sources,
                           strip_audio_command, verified_clips)

SHARES = {"F1": 0.35, "F2": 0.35, "F3": 0.15, "F4": 0.15}


@pytest.fixture(scope="module")
def items():
    return build_split(make_clips(), "test", 4000, make_disjoint(), random.Random(0))


def of(items, **kw):
    return [i for i in items if all(getattr(i, k) == v for k, v in kw.items())]


def test_family_shares(items):
    counts = Counter(i.family for i in items)
    for family, share in SHARES.items():
        assert counts[family] / len(items) == pytest.approx(share, abs=0.001)


def test_half_aligned(items):
    assert len(of(items, kind=ALIGNED)) == len(items) / 2


def test_swap_directions_have_equal_shares(items):
    audio_side = len(of(items, kind=AUDIO_SWAP)) + len(of(items, kind=SILENCE))
    assert audio_side == pytest.approx(len(of(items, kind=VIDEO_SWAP)), abs=2)


def test_silence_is_f1_only_and_20_percent_of_f1(items):
    silence = of(items, kind=SILENCE)
    assert {i.family for i in silence} == {"F1"}
    assert len(silence) / len(of(items, family="F1")) == pytest.approx(0.2, abs=0.002)
    assert all(i.audio_clip is None and i.audio_label is None and i.answer == "no" for i in silence)
    assert all(i.follows_A == "no" and i.follows_V == "yes" for i in silence)


@pytest.mark.parametrize("family", ["F1", "F2", "F3"])
def test_yes_no_balance(items, family):
    fam = of(items, family=family)
    assert Counter(i.answer for i in fam)["yes"] / len(fam) == pytest.approx(0.5, abs=0.002)


@pytest.mark.parametrize("family", ["F1", "F2"])
def test_yes_no_balance_within_conflict(items, family):
    conflict = [i for i in of(items, family=family) if i.kind != ALIGNED]
    assert Counter(i.answer for i in conflict)["yes"] / len(conflict) == pytest.approx(0.5, abs=0.01)


def test_f4_roles_25_percent_and_shuffled(items):
    f4 = of(items, family="F4")
    roles = Counter(i.answer_class for i in f4)
    assert set(roles) == {"correct_audio", "correct_visual", "both", "neither"}
    assert all(c / len(f4) == pytest.approx(0.25, abs=0.01) for c in roles.values())
    assert len({i.options for i in f4}) > 1
    assert all(sorted(i.options) == sorted(F4_OPTIONS) for i in f4)


def test_twins_share_template_question_source(items):
    groups = defaultdict(list)
    for item in items:
        groups[item.twin_id].append(item)
    assert all(len(g) == 2 for g in groups.values())
    for a, b in groups.values():
        x, y = (a, b) if a.kind == ALIGNED else (b, a)
        assert x.kind == ALIGNED and y.kind != ALIGNED
        assert x.family == y.family and x.question == y.question and x.options == y.options
        assert x.video_clip == x.audio_clip
        assert x.video_clip in (y.video_clip, y.audio_clip)
        assert x.answer != y.answer


def test_answers_match_labels(items):
    for i in items:
        if i.family == "F1":
            assert i.answer == ("yes" if i.cls == i.audio_label else "no")
        elif i.family == "F2":
            assert i.answer == ("yes" if i.cls == i.visual_label else "no")
        elif i.family == "F3":
            assert i.question == F3_T and i.answer == ("yes" if i.audio_label == i.visual_label else "no")
        else:
            heard, seen = i.cls == i.audio_label, i.cls == i.visual_label
            assert i.answer == ("both" if heard and seen else "audio" if heard else "video" if seen else "neither")


def test_swap_conflict_structure(items):
    for i in of(items, kind=AUDIO_SWAP) + of(items, kind=VIDEO_SWAP):
        assert i.audio_label != i.visual_label


def test_f1_audio_swap_semantics(items):
    for i in of(items, kind=AUDIO_SWAP, family="F1"):
        if i.answer == "yes":
            assert i.follows_A == "yes" and i.follows_V == "no"
        else:
            assert i.follows_A == "no" and i.follows_V == "yes"


def test_f2_video_swap_follows_visual_truth(items):
    for i in of(items, kind=VIDEO_SWAP, family="F2"):
        assert i.answer == i.follows_V != i.follows_A


def test_f3_null_follows(items):
    assert all(i.follows_A is None and i.follows_V is None for i in of(items, family="F3"))


def test_f4_follows(items):
    for i in of(items, family="F4"):
        assert i.follows_A == ("audio" if i.cls == i.audio_label else "neither")
        assert i.follows_V == ("video" if i.cls == i.visual_label else "neither")


def test_answer_in_options_for_mcq(items):
    assert all(i.answer in i.options for i in of(items, family="F4"))


def test_label_disjoint_only():
    disjoint = {"a": {"b"}, "b": {"a"}}
    clips = [Clip(f"{c}{i}", f"s{c}{i}", c) for c in "abc" for i in range(5)]
    built = build_split(clips, "t", 40, disjoint, random.Random(0), shares={"F1": 1.0}, silence_share=0)
    assert {i.audio_label for i in built} | {i.visual_label for i in built} <= {"a", "b"}


def test_similarity_tier_bottom_half():
    sim = {frozenset(p): s for p, s in [(("a", "b"), 0.9), (("a", "c"), 0.1), (("b", "c"), 0.2), (("c", "d"), 0.95)]}
    assert label_disjoint_pairs(sim, 0.5) == {"a": {"c"}, "c": {"a", "b"}, "b": {"c"}}


def test_partner_must_score_low_visually():
    source = Clip("x", "s1", "a", partner_scores={"b": 0.9})
    partner = Clip("y", "s2", "b")
    with pytest.raises(ValueError):
        build_split([source, partner], "t", 2, {"a": {"b"}, "b": {"a"}}, random.Random(0), shares={"F2": 1.0}, silence_share=0, threshold=0.5)
    source.partner_scores = {"b": 0.1}
    assert build_split([source, partner], "t", 2, {"a": {"b"}, "b": {"a"}}, random.Random(0), shares={"F2": 1.0}, silence_share=0, threshold=0.5)


def test_verified_filter():
    clips = [Clip("a", "s", "dog", 0.9), Clip("b", "s", "dog", 0.1), Clip("c", "s", "cat", 0.9)]
    assert [c.clip_id for c in verified_clips(clips, 0.5, ["cat"])] == ["a"]


def test_split_sources_are_disjoint_and_whole():
    clips = [Clip(f"c{i}", f"s{i // 3}", "dog") for i in range(300)]
    splits = split_sources(clips, {"a": 100, "b": 100, "c": 200}, seed=0, exclude_sources=["s0"])
    sources = [{c.source_video for c in v} for v in splits.values()]
    assert not sources[0] & sources[1] and not sources[1] & sources[2] and not sources[0] & sources[2]
    assert "s0" not in set.union(*sources)
    assert all(len([c for c in clips if c.source_video == s]) == 3 for s in sources[0])
    assert sum(len(v) for v in splits.values()) == 297
    assert len(splits["c"]) == pytest.approx(150, abs=6)


def test_xm_splits_source_disjoint(xm):
    assert_source_disjoint(xm)


def test_disjointness_check_catches_a_leak(xm):
    with pytest.raises(ValueError):
        assert_source_disjoint({"gate": xm["gate"], "copy": xm["gate"]})
    with pytest.raises(ValueError):
        assert_source_disjoint({"gate": xm["gate"]}, external_sources=list(xm["gate"][0].sources))


def test_split_sizes(xm):
    assert {k: len(v) for k, v in xm.items()} == {"train": 400, "guard": 1024, "dev": 512, "gate": 500, "test": 2000}


def test_gate_uses_f1_f2_without_silence(xm):
    assert {i.family for i in xm["gate"]} == {"F1", "F2"}
    assert not of(xm["gate"], kind=SILENCE)


def test_guard_planted_carriers_f1_f2_rest_at_shares(xm):
    carriers = [i for i in xm["guard"] if i.planted_carrier]
    rest = [i for i in xm["guard"] if not i.planted_carrier]
    assert len(carriers) == 256 and len(rest) == 768
    assert {i.family for i in carriers} == {"F1", "F2"} and not of(carriers, kind=SILENCE)
    counts = Counter(i.family for i in rest)
    assert counts["F3"] / 768 == pytest.approx(0.15, abs=0.005)
    assert of(rest, kind=SILENCE)


def test_guard_item_ids_unique(xm):
    assert len({i.item_id for i in xm["guard"]}) == 1024


def test_overlap_filter():
    clips = [Clip("a", "yt1", "dog"), Clip("b", "yt2", "dog"), Clip("c", "yt3", "dog")]
    kept, dropped = overlap_filter(clips, ["yt1", "c"])
    assert [c.clip_id for c in kept] == ["b"] and len(dropped) == 2


def test_load_clips(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text('clip_id,source_video,label,siglip_score,partner_scores\na,s,dog,0.7,"{""cat"": 0.1}"\n')
    assert load_clips(path)[0].partner_scores == {"cat": 0.1}


def test_sample_batch_respects_p():
    natural, xm = [Item("n", "q", "yes")], [Item("x", "q", "yes", is_xm=True)]
    picks = sample_batch(natural, xm, 8000, 0.3, random.Random(0))
    assert sum(i.is_xm for i in picks) / 8000 == pytest.approx(0.3, abs=0.02)
    assert not any(i.is_xm for i in sample_batch(natural, xm, 100, 0.0, random.Random(0)))


@pytest.fixture(scope="module")
def planted(xm):
    bases = {i.item_id: GOOD for i in xm["gate"]}
    return build_planted(xm["gate"], bases, make_disjoint(), random.Random(0), per_type=125)


def test_gate_type_counts(planted):
    assert len(planted) == 500
    assert Counter(p.ptype for p in planted) == {"false_audio": 125, "false_visual": 125, "modality_swapped": 125, "control": 125}


def test_gate_segments(planted):
    counts = Counter((p.ptype, p.segment) for p in planted)
    assert counts[("false_audio", "audio")] == 125 and counts[("false_visual", "visual")] == 125
    assert counts[("modality_swapped", "audio")] == 63 and counts[("modality_swapped", "visual")] == 62
    assert counts[("control", "audio")] == 63 and counts[("control", "visual")] == 62


def test_guard_planted_64_per_type(xm):
    carriers = [i for i in xm["guard"] if i.planted_carrier]
    built = build_planted(carriers, {i.item_id: GOOD for i in carriers}, make_disjoint(), random.Random(0), per_type=64)
    assert Counter(p.ptype for p in built) == {t: 64 for t in ("false_audio", "false_visual", "modality_swapped", "control")}
    assert Counter((p.ptype, p.segment) for p in built)[("control", "audio")] == 32
    assert all(p.item.planted_carrier for p in built)


def test_each_item_carries_one_planted_path(planted):
    assert len({p.item.item_id for p in planted}) == 500


def test_planted_sentence_lands_where_recorded(planted):
    for p in planted:
        item = Item("x", p.item.question, p.item.answer, is_xm=True)
        path = parse_path(p.planted_text, item)
        assert path.valid and getattr(path, p.segment)[p.index] == p.sentence


def test_insert_position_is_uniform(planted):
    positions = Counter(p.index for p in planted)
    assert len(positions) == 4 and min(positions.values()) > 70  # 3 base sentences, 4 positions


def test_planted_truth_by_type(planted):
    for p in planted:
        labels = (p.item.audio_label, p.item.visual_label)
        cls = p.sentence
        if p.ptype == "control":
            assert (p.item.audio_label if p.segment == "audio" else p.item.visual_label) in cls
        elif p.ptype == "modality_swapped":
            assert p.item.kind in (AUDIO_SWAP, VIDEO_SWAP)
            assert (p.item.visual_label if p.segment == "audio" else p.item.audio_label) in cls
        else:
            assert not any(label in cls for label in labels)


def test_false_class_is_label_disjoint(xm):
    disjoint = {"dog": {"piano"}, "piano": {"dog"}, "cat": {"piano"}}
    item = next(i for i in xm["gate"] if i.kind == ALIGNED)
    item = type(item)(**{**item.__dict__, "audio_label": "dog", "visual_label": "dog"})
    from avrsi.perturb import _false_classes
    assert _false_classes(item, disjoint) == ["piano"]


def test_shared_style_pool_for_true_and_false(planted):
    strip = lambda p: p.sentence.replace(p.item.audio_label or "", "{c}").replace(p.item.visual_label, "{c}")  # noqa: E731
    from avrsi.perturb import AUDIO_STYLES, VISUAL_STYLES
    assert {strip(p) for p in planted if p.ptype == "control"} == set(AUDIO_STYLES) | set(VISUAL_STYLES)


def test_redraw_when_dedupe_would_remove(xm):
    from avrsi.perturb import _plant
    item = xm["gate"][0]
    base = "<audio>The sound of rain can be heard.</audio><visual>x.</visual><fusion></fusion><answer>yes</answer>"
    built = _plant(item, base, "false_audio", "audio", ["rain", "drum"], random.Random(0))
    assert "The sound of rain" not in built.sentence or built.index == 0
    # Whatever is drawn, the kept planted sentence is the one the path records.
    for seed in range(30):
        built = _plant(item, base, "false_audio", "audio", ["rain"], random.Random(seed))
        path = parse_path(built.planted_text, Item("x", "q", "yes", is_xm=True))
        assert path.audio[built.index] == built.sentence


def test_planted_redraw_exhausts_only_when_nothing_survives(xm, monkeypatch):
    import avrsi.perturb as perturb
    monkeypatch.setattr(perturb, "dedupe_indices", lambda sentences: [])
    with pytest.raises(ValueError, match="deduped"):
        perturb._plant(xm["gate"][0], GOOD, "control", "audio", ["rain"], random.Random(0))


def test_silence_excluded_from_planted(xm):
    items = [i for i in xm["test"] if i.family in ("F1", "F2")]
    built = build_planted(items, {i.item_id: GOOD for i in items}, make_disjoint(), random.Random(0), per_type=10)
    assert not any(p.item.kind == SILENCE for p in built)


def test_invalid_base_skipped_and_shortage_raises(xm):
    with pytest.raises(ValueError):
        build_planted(xm["gate"], {xm["gate"][0].item_id: GOOD}, make_disjoint(), random.Random(0), per_type=125)


def test_base_selection_first_valid_draw():
    bases, report = select_base_paths({"a": ["bad", GOOD], "b": ["bad", "worse"], "c": [GOOD, "bad"]})
    assert set(bases) == {"a", "c"} and report["mode"] == "theta0" and not report["use_fallback_reference"]


def test_20_percent_rule_exactly_at_boundary_stays_theta0():
    assert select_base_paths({"a": [GOOD, "x", "x", "x", "x"]})[1]["mode"] == "theta0"


def test_20_percent_rule_uses_exemplar_and_fallback_reference():
    exemplar = {"a": "chat <audio>A dog.</audio> <visual>A man.</visual> <fusion>One. Two. Three.</fusion> <answer>yes</answer> bye"}
    bases, report = select_base_paths({"a": ["bad"] * 9 + [GOOD] * 0, "b": ["bad"]}, exemplar)
    assert report["mode"] == "exemplar" and report["use_fallback_reference"]
    assert valid_format(bases["a"]) and "Three" not in bases["a"]


def test_20_percent_rule_requires_exemplar():
    with pytest.raises(ValueError):
        select_base_paths({"a": ["bad"]})


def test_assemble_needs_both_segments():
    assert assemble_base("<audio>A dog.</audio>") is None


def test_mux_command_swaps_streams():
    cmd = mux_command("v.mp4", "a.mp4", "o.mp4")
    assert cmd[0] == "ffmpeg" and cmd[cmd.index("-map") + 1] == "0:v:0" and "1:a:0" in cmd and cmd[-1] == "o.mp4"
    assert cmd.index("v.mp4") < cmd.index("a.mp4")


def test_mux_silence_replaces_whole_track():
    cmd = mux_command("v.mp4", None, "o.mp4")
    assert any("anullsrc" in c for c in cmd) and "1:a:0" in cmd and "0:a:0" not in cmd


def test_strip_and_extract_commands():
    assert "-an" in strip_audio_command("v", "o") and "0:a:0" not in strip_audio_command("v", "o")
    assert "-vn" in extract_audio_command("v", "o")


def test_mux_item_invokes_runner(tmp_path, items):
    calls = []
    swap = of(items, kind=AUDIO_SWAP)[0]
    out = mux_item(swap, lambda c: f"/clips/{c}.mp4", tmp_path, runner=lambda cmd, check: calls.append((cmd, check)))
    assert out.name == f"{swap.item_id}.mp4"
    assert f"/clips/{swap.video_clip}.mp4" in calls[0][0] and f"/clips/{swap.audio_clip}.mp4" in calls[0][0] and calls[0][1]


def test_freeze_hash_and_no_overwrite(tmp_path, items):
    path = tmp_path / "test.jsonl"
    digest = freeze(items[:6], path)
    assert (tmp_path / "test.jsonl.sha256").read_text().strip() == digest
    assert load_frozen(path)[0].sources == tuple(sorted(items[:6], key=lambda i: i.item_id)[0].sources)
    with pytest.raises(FileExistsError):
        freeze(items[:6], path)


def test_frozen_edit_is_detected(tmp_path, items):
    path = tmp_path / "test.jsonl"
    freeze(items[:6], path)
    row = json.loads(path.read_text().splitlines()[0])
    path.write_text(json.dumps({**row, "answer": "flipped"}) + "\n")
    with pytest.raises(ValueError):
        load_frozen(path)


def test_planted_roundtrip(tmp_path, planted):
    save_planted(planted[:3], {"mode": "theta0"}, tmp_path / "p.json")
    loaded, report = load_planted(tmp_path / "p.json")
    assert loaded[0].planted_text == planted[0].planted_text and loaded[0].item.sources == planted[0].item.sources
    assert report == {"mode": "theta0"}


def test_human_check():
    assert human_check_passes(25) and not human_check_passes(26)
