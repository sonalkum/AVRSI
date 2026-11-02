"""scripts/smoke_qwen3.py on the tiny real thinker, the real processor and generated clips."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("peft")
from qwen3_tiny import BASE, tiny_thinker  # noqa: E402
from test_qwen3_preprocess import SNAPSHOT, make_clip  # noqa: E402
from transformers import Qwen3OmniMoeProcessor  # noqa: E402

from avrsi.models.qwen import Qwen3OmniBackend  # noqa: E402

pytestmark = pytest.mark.skipif(not (SNAPSHOT / "vocab.json").exists(), reason="snapshot with tokenizer files absent")
ROOT = Path(__file__).resolve().parent.parent


def load_smoke():
    spec = importlib.util.spec_from_file_location("smoke_qwen3", ROOT / "scripts" / "smoke_qwen3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tiny(Qwen3OmniBackend):
    def _load(self):
        return Qwen3OmniMoeProcessor.from_pretrained(SNAPSHOT), tiny_thinker()


def make_tiny(args, device, gpus):
    return Tiny({**BASE, "num_frames": 4, "lora_exclude_modules": r"(audio_tower|visual)\..*", "gradient_checkpointing": True, "logprob_batch": 2,
                 "generate_batch": 4}, device="cpu")


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("vgg")
    ids = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd"]
    for i in ids:
        make_clip(root / f"{i}_000001.mp4")
    labels = root / "labels.json"
    labels.write_text(json.dumps({i: {"vgg": [f"the sound of class {i[0]}"]} for i in ids}))
    return root, labels


def test_two_a8_steps_with_a_critic_handoff(clips, tmp_path):
    root, labels = clips
    report = load_smoke().main(["--clips", "4", "--batch", "2", "--clip-seconds", "2", "--clips-dir", str(root), "--labels", str(labels), "--max-new-tokens", "6",
                                "--critic-max-new-tokens", "3", "--probe-length", "4", "--out", str(tmp_path)], make_tiny)
    assert report["rollouts"] > 0 and report["forced_paths"] > 0
    assert len(report["step_seconds"]) == 2 and report["rewards"]["trajectories"] == [8, 8]
    assert report["rewards"]["malformed_rate"] == [0.0, 0.0]  # forced, so critics run
    assert all(p > 0 for p in report["rewards"]["passes"])
    assert report["guard"][0]["step"] == 2 and report["critic_sync_ran"] == report["guard"][0]["handoff"]
    assert (tmp_path / "smoke.json").exists() and (tmp_path / "run" / "checkpoints" / "LATEST").exists()
    assert report["lora_tensors"] == 16


def test_forced_format_leaves_critics_and_valid_paths_alone():
    smoke = load_smoke()
    from avrsi.models.base import CRITIC, Generation, ModelInput, Sampling

    class Plain:
        def generate(self, inputs, sampling, role):
            return [Generation("<answer>yes</answer>", [1]), Generation("rambling", [2])]

        def encode(self, text):
            return [3]

    forced = smoke.ForcedFormat(Plain())
    x = ModelInput("Answer the question about this clip.")
    out = forced.generate([x, x], Sampling(1.0, 1.0, True, 4))
    assert out[0].token_ids == [1] and out[1].token_ids == [3] and forced.forced == 1 and forced.rollouts == 2
    assert forced.generate([x, x], Sampling(1.0, 1.0, True, 4), CRITIC)[1].text == "rambling"
