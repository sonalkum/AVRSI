import json

import pytest

from avrsi.data import REGISTRY, load_benchmark
from avrsi.data.avhbench import load_avhbench
from avrsi.data.cmm import load_cmm
from avrsi.models.base import build_model_input

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


@pytest.mark.parametrize("name", sorted(set(REGISTRY) - {"avhbench", "cmm"}))
def test_other_benchmarks_name_the_verify(name):
    with pytest.raises(NotImplementedError, match=r"\[VERIFY\]"):
        load_benchmark(name, "path")


def test_unknown_benchmark():
    with pytest.raises(KeyError):
        load_benchmark("nope")
