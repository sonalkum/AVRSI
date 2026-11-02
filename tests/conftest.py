import random
import sys
from pathlib import Path

import pytest

RUN_LIB = Path(__file__).resolve().parent.parent / ".venv" / "run_lib"
if RUN_LIB.is_dir():  # peft, av, imageio-ffmpeg; see requirements-qwen3.txt
    sys.path.append(str(RUN_LIB))

from avrsi.config import resolve_arm
from avrsi.models.base import Sampling
from avrsi.parse import Item
from avrsi.perturb import Clip, build_xm, label_disjoint_pairs, split_sources
from avrsi.train import Samplings

GOOD = "<audio>A dog barks. A car passes. Rain falls.</audio><visual>A man walks. A tree sways. A bird flies.</visual><fusion>They agree.</fusion><answer>Yes</answer>"
CLASSES = ["dog", "piano", "rain", "siren", "drum", "cat"]


@pytest.fixture
def good():
    return GOOD


@pytest.fixture
def yes_item():
    return Item("i1", "Is a dog in the clip?", "yes")


@pytest.fixture
def mcq_item():
    return Item("i2", "Where is rain present?", "video", options=("neither", "video", "both", "audio"), is_xm=True)


@pytest.fixture
def rng():
    return random.Random(0)


@pytest.fixture
def a8():
    return resolve_arm("A8")


@pytest.fixture
def samplings():
    return Samplings(Sampling(1.0, 1.0, True, 64), Sampling(0.0, 1.0, False, 64), Sampling(0.0, 1.0, False, 64))


def make_clips(n=600, classes=CLASSES):
    return [Clip(f"clip{i}", f"src{i}", classes[i % len(classes)], partner_scores=dict.fromkeys(classes, 0.0)) for i in range(n)]


def make_disjoint(classes=CLASSES):
    similarity = {frozenset((a, b)): 0.0 for i, a in enumerate(classes) for b in classes[i + 1:]}
    return label_disjoint_pairs(similarity, tier=1.0)


@pytest.fixture(scope="session")
def xm():
    sizes = {"train": 400, "guard": 1024, "dev": 512, "gate": 500, "test": 2000}
    clips = make_clips(1500)
    splits = split_sources(clips, sizes, seed=1)
    return build_xm(splits, make_disjoint(), seed=1, sizes=sizes)
