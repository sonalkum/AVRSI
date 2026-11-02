import json

import numpy as np
import pytest

from avrsi.labels import class_similarity, cosine_similarity, max_frame_scores, score_clip, uniform_frame_indices, write_similarity
from avrsi.perturb import _visually_absent


def test_uniform_frames():
    assert uniform_frame_indices(100, 8) == [0, 14, 28, 42, 57, 71, 85, 99]
    assert len(uniform_frame_indices(3)) == 8


def test_max_over_frames():
    assert max_frame_scores(np.array([[0.1, 0.9], [0.8, 0.2]])).tolist() == [0.8, 0.9]


def test_score_clip_fills_label_and_partners():
    scorer = lambda frames, classes: np.array([[0.9, 0.1, 0.3], [0.2, 0.05, 0.4]])  # noqa: E731
    clip = score_clip("c", "s", "dog", ["f1", "f2"], ["dog", "cat", "car"], scorer)
    assert clip.siglip_score == pytest.approx(0.9) and set(clip.partner_scores) == {"cat", "car"}
    assert _visually_absent(clip, "cat", 0.5) and clip.partner_scores["car"] == pytest.approx(0.4)


def test_similarity_takes_the_closer_modality():
    clap = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]), "c": np.array([0.0, 2.0])}
    siglip = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]), "c": np.array([0.0, 1.0])}
    assert cosine_similarity(clap)[frozenset("ab")] == pytest.approx(1.0)
    table = class_similarity(clap, siglip)
    assert table[frozenset("ab")] == pytest.approx(1.0) and table[frozenset("bc")] == pytest.approx(1.0)
    assert table[frozenset("ac")] == pytest.approx(0.0)


def test_similarity_file_matches_build_xm(tmp_path):
    path = tmp_path / "s.json"
    write_similarity({frozenset(("b", "a")): 0.5}, path)
    assert json.loads(path.read_text()) == {"a|b": 0.5}
