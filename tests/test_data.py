import json

import pytest

from avrsi.data import REGISTRY, load_benchmark
from avrsi.data.avhbench import load_avhbench
from avrsi.data.avqa import load_avqa
from avrsi.data.cmm import load_cmm
from avrsi.models.base import build_model_input
from avrsi.parse import parse_path, question_prompt
from avrsi.rewards import r_outcome

GOOD = "<audio>A scream.</audio><visual>A ride.</visual><fusion>They agree.</fusion><answer>{}</answer>"
AVH = [
    {"video_id": "00191", "task": "Video-driven Audio Hallucination", "text": "Is the couch making sound in the audio?", "label": "No"},
    {"video_id": "00191", "task": "Audio-driven Video Hallucination", "text": "Is the man visible in the video?", "label": "Yes"},
    {"video_id": "00191", "task": "AV Matching", "text": "Are they matching?", "label": "Yes"},
    {"video_id": "00191", "task": "AV Captioning", "text": "Describe it.", "label": "A man snores."},
]
CMM = [
    {"category": "x", "sub_category": "visual-language", "modality": "visual", "granularity": "o", "correlation_type": "g",
     "video_path": "./raw/a.mp4", "audio_path": None, "question": "Did you see river in the video?", "answer": "yes"},
    {"category": "x", "sub_category": "visual-language", "modality": "visual", "granularity": "o", "correlation_type": "g",
     "video_path": "./raw/b.mp4", "audio_path": None, "question": "Did you see tree in the video?", "answer": "no"},
    {"category": "x", "sub_category": "audio-language", "modality": "audio", "granularity": "o", "correlation_type": "g",
     "video_path": None, "audio_path": "./raw/c.wav", "question": "Did you hear a dog in the audio?", "answer": "Yes"},
    {"category": "y", "sub_category": "overrely_visual_ignore_audio", "modality": "visual+audio", "granularity": "o", "correlation_type": "g",
     "video_path": "./raw/d.mp4", "audio_path": "./raw/d.wav", "question": "Did you hear bird in the audio?", "answer": "no"},
]


@pytest.fixture
def avh(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(AVH))
    return path


@pytest.fixture
def cmm(tmp_path):
    path = tmp_path / "cmm.json"
    path.write_text(json.dumps(CMM))
    return path


def test_avhbench_fields(avh):
    items = load_avhbench(avh, "/videos")
    assert [i.meta["task"] for i in items] == ["v2a", "a2v", "matching", "captioning"]
    assert items[0].answer == "no" and items[0].video == "/videos/00191.mp4" and items[3].answer == "A man snores."
    assert not items[0].is_xm


def test_avhbench_task_filter(avh):
    assert len(load_avhbench(avh, tasks=("a2v", "v2a"))) == 2


def test_avhbench_rejects_unknown_task(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{**AVH[0], "task": "other"}]))
    with pytest.raises(ValueError):
        load_avhbench(path)


def test_avhbench_rejects_bad_label(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{**AVH[0], "label": "maybe"}]))
    with pytest.raises(ValueError):
        load_avhbench(path)


def test_cmm_modalities_are_exactly_what_the_sample_has(cmm):
    items = load_cmm(cmm, "/root")
    assert [i.modalities for i in items] == [("video",), ("video",), ("audio",), ("audio", "video")]
    assert items[2].video is None and items[2].audio == "/root/raw/c.wav"
    assert items[2].answer == "yes"


def test_cmm_video_only_never_sends_audio(cmm):
    item = load_cmm(cmm)[0]
    assert build_model_input(item, ("audio", "video"), "p").audio is None


def test_cmm_clusters_and_subcategories(cmm):
    items = load_cmm(cmm)
    assert [i.meta["sample_id"] for i in items] == [0, 0, 1, 1]
    assert items[3].meta["subcategory"] == "Visual Dominance"
    assert [i.meta["sample_id"] for i in load_cmm(cmm, sample_ids=[9, 9, 8, 8])] == [9, 9, 8, 8]


def test_cmm_rejects_unknown_subcategory(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps([{**CMM[0], "sub_category": "new"}]))
    with pytest.raises(ValueError):
        load_cmm(path)


AVQA = {"id": 1205, "video_name": "0AEJTlHIhz0_000358", "video_id": 1467, "question_text": "Why do the people in the video scream?",
        "multi_choice": ["Roller coaster", "On a pirate ship", "Take Ferris wheel", "Take the jumping machine"], "answer": 0,
        "question_relation": "Both", "question_type": "Why"}  # the example on the dataset page


@pytest.fixture
def avqa(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "0AEJTlHIhz0_000358.mp4").touch()
    (tmp_path / "train_qa.json").write_text(json.dumps([AVQA, {**AVQA, "id": 2, "video_name": "gone_000000"}]))
    (tmp_path / "val_qa.json").write_text(json.dumps([{**AVQA, "id": 7, "answer": 3}]))
    return tmp_path


def test_avqa_documented_row(avqa):
    item = load_avqa(avqa)[0]
    assert (item.item_id, item.answer) == ("avqa-1205", "Roller coaster") and len(item.options) == 4
    assert item.video == str(avqa / "videos" / "0AEJTlHIhz0_000358.mp4") and item.audio is None
    assert item.modalities == ("audio", "video") and not item.is_xm
    assert item.meta["relation"] == "Both" and item.meta["question_type"] == "Why" and item.meta["video_id"] == 1467


def test_avqa_is_scored_by_option_and_prompted_with_choices(avqa):
    item = load_avqa(avqa)[0]
    assert "A. Roller coaster" in question_prompt(item) and "D. Take the jumping machine" in question_prompt(item)
    for answer in ("A", "Roller coaster"):
        assert r_outcome(parse_path(GOOD.format(answer), item), item) == 1
    assert r_outcome(parse_path(GOOD.format("B"), item), item) == 0


def test_avqa_audio_comes_from_the_clip_and_can_be_withheld(avqa):
    item = load_avqa(avqa)[0]
    assert build_model_input(item, ("audio",), "p").audio == item.video
    assert build_model_input(item, ("video",), "p").audio is None


def test_avqa_registered_for_training_and_h5(avqa, capsys):
    assert [i.item_id for i in load_benchmark("avqa", avqa)] == ["avqa-1205"]
    assert "dropped 1 of 2" in capsys.readouterr().out
    assert load_benchmark("avqa_test", avqa)[0].answer == "Take the jumping machine"


@pytest.mark.parametrize("change", [{"answer": 4}, {"answer": -1}, {"question_relation": "Smell"}])
def test_avqa_rejects_bad_rows(avqa, change):
    (avqa / "train_qa.json").write_text(json.dumps([{**AVQA, **change}]))
    with pytest.raises(ValueError):
        load_avqa(avqa, check_media=False)


@pytest.mark.parametrize("name", sorted(set(REGISTRY) - {"avhbench", "cmm", "avqa", "avqa_test"}))
def test_other_benchmarks_name_the_verify(name):
    with pytest.raises(NotImplementedError, match=r"\[VERIFY\]"):
        load_benchmark(name, "path")


def test_unknown_benchmark():
    with pytest.raises(KeyError):
        load_benchmark("nope")
