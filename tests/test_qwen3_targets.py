"""LoRA targets and checkpoint layout against the real config, on the meta device. Skipped where the
snapshot is not on disk. Set AVRSI_QWEN3_SNAPSHOT to another copy of it."""
import json
import math
import os
import re
import struct
from pathlib import Path

import pytest
import torch

peft = pytest.importorskip("peft")
from transformers import Qwen3OmniMoeThinkerForConditionalGeneration as Thinker  # noqa: E402
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig  # noqa: E402

from avrsi.config import load_model  # noqa: E402

HUB = Path("/fs/gamma-projects/audio/.cache/hub")
INSTRUCT = Path(os.environ.get("AVRSI_QWEN3_SNAPSHOT", HUB / "models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"))
CAPTIONER = HUB / "models--Qwen--Qwen3-Omni-30B-A3B-Captioner/snapshots/a2bd106cbf527db5676e79662674da22b0545ec0"
LAYERS = 48
pytestmark = pytest.mark.skipif(not (INSTRUCT / "config.json").exists(), reason="Instruct snapshot config absent")


@pytest.fixture(scope="module")
def settings():
    return load_model("qwen3_omni")


def meta_thinker():
    with torch.device("meta"):
        return Thinker(Qwen3OmniMoeThinkerConfig.from_pretrained(INSTRUCT))


@pytest.fixture(scope="module")
def thinker():
    return meta_thinker()


@pytest.fixture(scope="module")
def peft_model(settings):
    config = peft.LoraConfig(r=32, lora_alpha=settings["lora_alpha"], target_modules=settings["lora_target_modules"],
                             exclude_modules=settings["lora_exclude_modules"])
    return peft.get_peft_model(meta_thinker(), config, adapter_name="actor", low_cpu_mem_usage=True)


def test_regex_hits_exactly_the_decoder_attention(thinker, settings):
    pattern = settings["lora_target_modules"]
    hits = [n for n, _ in thinker.named_modules() if re.fullmatch(pattern, n)]
    assert len(hits) == 4 * LAYERS == 192
    assert all(isinstance(thinker.get_submodule(n), torch.nn.Linear) for n in hits)
    assert not [n for n in hits if n.startswith(("audio_tower", "visual", "lm_head")) or "mlp" in n]


def test_experts_cannot_be_lora_targets(thinker):
    """They are raw parameters, one tensor per layer holding all 128 experts."""
    experts = thinker.model.layers[0].mlp.experts
    assert not list(experts.children()) or all(not isinstance(m, torch.nn.Linear) for m in experts.children())
    assert experts.gate_up_proj.shape[0] == 128


def test_peft_alone_would_widen_the_regex_to_the_audio_tower(thinker, settings):
    """Why lora_exclude_modules exists. If this starts failing, PEFT stopped rewriting the regex."""
    config = peft.LoraConfig(r=2, target_modules=settings["lora_target_modules"])
    model = peft.get_peft_model(meta_thinker(), config, adapter_name="actor", low_cpu_mem_usage=True)
    assert any(n.startswith("audio_tower") for n in model.base_model.targeted_module_names)


def test_real_peft_injection_stays_out_of_the_towers(peft_model):
    names = peft_model.base_model.targeted_module_names
    assert len(names) == 192
    assert all(re.fullmatch(r"model\.layers\.\d+\.self_attn\.[qkvo]_proj", n) for n in names)


def test_trainable_parameter_count(peft_model):
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    per_layer = 32 * ((2048 + 4096) + 2 * (2048 + 512) + (4096 + 2048))
    assert trainable == 48 * per_layer == 26_738_688
    assert not [n for n, p in peft_model.named_parameters() if p.requires_grad and "lora_" not in n]


def header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size))


EXPERT = re.compile(r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")


def fused(key):
    """The hub stores one Linear per expert; transformers 5 fuses them on load."""
    key = key.removeprefix("thinker.")
    match = EXPERT.fullmatch(key)
    if match is None:
        return key
    return f"{match[1]}.{'down_proj' if match[3] == 'down_proj' else 'gate_up_proj'}"


@pytest.mark.parametrize("snapshot", [INSTRUCT, CAPTIONER])
def test_every_thinker_key_is_in_the_checkpoint(thinker, snapshot):
    keys = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(thinker.state_dict()) == {fused(k) for k in keys if k.startswith("thinker.")}


def test_checkpoint_shapes_match_the_class(thinker):
    """Tensor shapes from safetensors headers alone; no weights are read."""
    shards = sorted(CAPTIONER.glob("model-*.safetensors"))
    if len(shards) != 16:
        pytest.skip("Captioner weights absent")
    state = thinker.state_dict()
    elements = 0
    for shard in shards:
        for key, meta in header(shard).items():
            if key == "__metadata__":
                continue
            elements += math.prod(meta["shape"])
            name = key.removeprefix("thinker.")
            if EXPERT.fullmatch(name):
                continue
            assert list(state[name].shape) == meta["shape"], key
    assert elements == sum(v.numel() for v in state.values())
    assert elements * 2 == 63_438_410_976  # bytes at bf16


def test_balanced_map_fits_two_48gb_gpus_and_not_one(thinker):
    """Why a rank may span GPUs: 63.4 GB of weights does not fit an RTX A6000."""
    from accelerate import infer_auto_device_map

    limits = {i: 40 * 2**30 for i in range(2)}
    keep_whole = list(Thinker._no_split_modules)
    placed = infer_auto_device_map(thinker, max_memory=limits, no_split_module_classes=keep_whole, dtype=torch.bfloat16)
    assert set(placed.values()) == {0, 1}
    spilled = infer_auto_device_map(thinker, max_memory={0: limits[0]}, no_split_module_classes=keep_whole, dtype=torch.bfloat16)
    assert set(spilled.values()) - {0}
