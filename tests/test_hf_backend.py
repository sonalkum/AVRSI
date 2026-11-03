"""HFBackend through a stub PEFT-shaped model, and the placeholders against the real token replacement."""
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

from avrsi.config import load_model
from avrsi.models.base import ACTOR, CRITIC, REFERENCE, Generation, ModelInput, Sampling
from avrsi.models import hf_common
from avrsi.models.hf_common import HFBackend, media_kwargs, messages, model_dtype, processor_kwargs

V = 6
PROMPT, NEW = [1, 2], [3, 4]
FLAG = "use_audio_in_video"
GREEDY = Sampling(0.0, 1.0, False, 4)
SAMPLED = Sampling(1.0, 1.0, True, 4)


class Encoded(dict):
    def to(self, target):
        if isinstance(target, torch.dtype):
            self["cast_to"] = target
        return self


class StubProcessor:
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: "text", encode=lambda text, add_special_tokens: [1])

    def __init__(self):
        self.last = None

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        return "prompt"

    def __call__(self, **kwargs):
        self.last = kwargs
        return Encoded(input_ids=torch.tensor([PROMPT]), attention_mask=torch.ones(1, 2, dtype=torch.long))


class StubModel(torch.nn.Module):
    """Logits are a base table plus the active adapter's table, indexed by the previous token."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.register_buffer("base", torch.randn(V, V, generator=generator))
        embed = lambda: torch.nn.Embedding(V, V)  # noqa: E731
        self.layer = torch.nn.ModuleDict({"lora_A": torch.nn.ModuleDict({"actor": embed(), "critic": embed()})})
        self.active, self.disabled, self.generations = "actor", False, []

    def add_adapter(self, name, config):
        pass

    def set_adapter(self, name):
        self.active = name
        for key, module in self.layer["lora_A"].items():
            module.weight.requires_grad_(key == name)

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        yield
        self.disabled = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits = self.base[input_ids]
        if not self.disabled:
            logits = logits + self.layer["lora_A"][self.active](input_ids)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.generations.append(dict(kwargs, adapter=None if self.disabled else self.active, grad=torch.is_grad_enabled()))
        return torch.cat([input_ids, torch.tensor([NEW])], 1)


class Stubbed(HFBackend):
    def _load(self):
        return StubProcessor(), StubModel()


CONFIG = {"audio_in_video_flag": FLAG, "missing_modality_fill": None, "lora_target_modules": ["layer"], "num_frames": None, "dtype": "bf16"}


def attached(monkeypatch):
    backend = Stubbed(dict(CONFIG), device="cpu")
    fake_peft = SimpleNamespace(LoraConfig=lambda **kwargs: kwargs, get_peft_model=lambda model, lora, adapter_name: model)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    backend.attach_lora(32)
    return backend


@pytest.fixture
def backend(monkeypatch):
    return attached(monkeypatch)


def weights(backend, name):
    return backend.model.layer["lora_A"][name].weight


def perturb_actor(backend):
    weights(backend, "actor").data.add_(1.0)


def test_sync_copies_actor_to_critic(backend):
    perturb_actor(backend)
    assert not torch.equal(weights(backend, "actor"), weights(backend, "critic"))
    backend.sync_critic()
    assert torch.equal(weights(backend, "actor"), weights(backend, "critic"))


def test_sync_does_not_alias(backend):
    backend.sync_critic()
    critic = weights(backend, "critic").detach().clone()
    perturb_actor(backend)
    assert torch.equal(weights(backend, "critic"), critic)


@pytest.mark.parametrize("role,adapter", [(ACTOR, "actor"), (CRITIC, "critic"), (REFERENCE, None)])
def test_generate_uses_the_role_then_restores_the_actor(role, adapter, backend):
    backend.generate([ModelInput("p")], GREEDY, role)
    assert backend.model.generations[-1]["adapter"] == adapter
    assert backend.model.active == "actor" and not backend.model.disabled and weights(backend, "actor").requires_grad


def test_score_restores_the_actor(backend):
    backend.score_logprobs([ModelInput("p")], [Generation("", NEW)], CRITIC)
    assert backend.model.active == "actor" and weights(backend, "actor").requires_grad


def expected_logps(table):
    logps = torch.log_softmax(table, -1)
    return torch.stack([logps[PROMPT[-1], NEW[0]], logps[NEW[0], NEW[1]]])


def test_logprobs_align_each_token_with_the_logits_before_it(backend):
    actor = backend.score_logprobs([ModelInput("p")], [Generation("", NEW)], ACTOR)[0]
    assert torch.allclose(actor, expected_logps(backend.model.base + weights(backend, "actor")))
    assert actor.requires_grad


def test_reference_is_the_base_with_adapters_off(backend):
    perturb_actor(backend)
    reference = backend.score_logprobs([ModelInput("p")], [Generation("", NEW)], REFERENCE)[0]
    assert torch.allclose(reference, expected_logps(backend.model.base))


def test_trainable_parameters_are_the_actors_only(backend):
    params = backend.trainable_parameters()
    assert len(params) == 1 and params[0] is weights(backend, "actor")


def test_generate_runs_without_gradients(backend):
    backend.generate([ModelInput("p")], GREEDY)
    assert backend.model.generations[-1]["grad"] is False


def test_generate_sets_sampling_explicitly(backend):
    backend.generate([ModelInput("p")], SAMPLED)
    sampled = backend.model.generations[-1]
    assert (sampled["top_k"], sampled["repetition_penalty"], sampled["do_sample"], sampled["top_p"]) == (0, 1.0, True, 1.0)
    backend.generate([ModelInput("p")], GREEDY)
    greedy = backend.model.generations[-1]
    assert greedy["repetition_penalty"] == 1.0 and greedy["do_sample"] is False and "top_k" not in greedy


@pytest.mark.parametrize("inputs,flag", [(ModelInput("p", video="c.mp4", audio="c.mp4"), True), (ModelInput("p", video="c.mp4"), False)])
def test_generate_passes_the_sampling_and_the_flag(backend, inputs, flag, monkeypatch):
    """Values off the generation_config defaults, so a dropped kwarg cannot match by accident."""
    monkeypatch.setattr(hf_common, "processor_kwargs", lambda *a, **k: processor_kwargs(*a, **k, audio_loader=str))
    monkeypatch.setattr(hf_common, "assert_withheld", lambda *a: None)  # the stub processor returns no media
    backend.generate([inputs], Sampling(0.7, 0.9, True, 4, top_k=5))
    sampled = backend.model.generations[-1]
    assert (sampled.get("temperature"), sampled.get("top_p"), sampled.get("top_k"), sampled.get(FLAG)) == (0.7, 0.9, 5, flag)


def test_bare_model_never_touches_adapters():
    """PreTrainedModel always has set_adapter, so before attach_lora the role must be a no-op."""
    backend = Stubbed(dict(CONFIG), device="cpu")

    def refuse(*args):
        raise AssertionError("no adapters loaded")

    backend.model.set_adapter = refuse
    backend.model.disable_adapter = refuse
    assert not backend._lora
    for role in (ACTOR, CRITIC, REFERENCE):
        backend.generate([ModelInput("p")], GREEDY, role)
        backend.score_logprobs([ModelInput("p")], [Generation("", NEW)], role)


def test_attach_marks_lora(backend):
    assert backend._lora


def test_avf_settings_reach_the_processor():
    config = load_model("avf")
    backend = Stubbed({**CONFIG, **config, "audio_in_video_flag": "load_audio_in_video"}, device="cpu")
    backend.generate([ModelInput("p")], GREEDY)
    last = backend.processor.last
    assert last["num_frames"] == 128 and last["size"] == {"height": 448, "width": 448} and last["audio_window_s"] == 30


def test_null_settings_stay_processor_defaults():
    assert media_kwargs(load_model("qwen2_5_omni")) == {}
    assert model_dtype(load_model("qwen3_omni")) == torch.bfloat16


def fake_self(cls):
    tokens = dict(audio_token="<A>", image_token="<I>", video_token="<V>", vision_bos_token="<vb>", vision_eos_token="<ve>",
                  audio_bos_token="<ab>", audio_eos_token="<ae>", image_processor=SimpleNamespace(merge_size=2),
                  video_processor=SimpleNamespace(merge_size=2))
    this = SimpleNamespace(**tokens)
    this.get_chunked_index = lambda *args: cls.get_chunked_index(this, *args)
    return this


def render(inputs):
    marks = {"audio": "<ab><A><ae>", "video": "<vb><V><ve>", "text": "q"}
    return "".join(marks[part["type"]] for part in messages(inputs)[0]["content"])


def replace_tokens(cls, inputs, audio_lengths):
    this = fake_self(cls)
    grids = [torch.tensor([2, 4, 4])] * inputs.modalities.count("video")
    flag = processor_kwargs(inputs, "t", FLAG, audio_loader=lambda path: path)[FLAG]
    text = cls.replace_multimodal_special_tokens(this, [render(inputs)], iter(audio_lengths), iter([]), iter(grids),
                                                 video_second_per_grid=iter([1.0]), use_audio_in_video=flag, position_id_per_seconds=25,
                                                 seconds_per_chunk=2.0)
    return text[0], flag


@pytest.mark.parametrize("cls", [Qwen2_5OmniProcessor, Qwen3OmniMoeProcessor])
def test_video_with_its_own_audio_consumes_one_length(cls):
    text, flag = replace_tokens(cls, ModelInput("p", video="c.mp4", audio="c.mp4"), [7])
    assert flag is True and text.count("<A>") == 7 and text.count("<V>") == 8 and "<ab><A><ae>" not in text


@pytest.mark.parametrize("cls", [Qwen2_5OmniProcessor, Qwen3OmniMoeProcessor])
def test_separate_audio_and_video_keep_both_placeholders(cls):
    text, flag = replace_tokens(cls, ModelInput("p", video="c.mp4", audio="c.wav"), [7])
    assert flag is False and text.count("<A>") == 7 and text.count("<V>") == 8


@pytest.mark.parametrize("cls", [Qwen2_5OmniProcessor, Qwen3OmniMoeProcessor])
def test_unimodal_inputs_replace_cleanly(cls):
    assert replace_tokens(cls, ModelInput("p", video="c.mp4"), [])[0].count("<V>") == 8
    assert replace_tokens(cls, ModelInput("p", audio="c.wav"), [7])[0].count("<A>") == 7


def test_old_placeholders_would_crash():
    """The bug: an audio placeholder plus the flag needs two lengths and the processor has one."""
    both = "<ab><A><ae><vb><V><ve>q"
    this = fake_self(Qwen2_5OmniProcessor)
    with pytest.raises(StopIteration):
        Qwen2_5OmniProcessor.replace_multimodal_special_tokens(
            this, [both], iter([7]), iter([]), iter([torch.tensor([2, 4, 4])]), video_second_per_grid=iter([1.0]), use_audio_in_video=True,
            position_id_per_seconds=25, seconds_per_chunk=2.0)


def test_encoded_inputs_are_cast_to_the_model_dtype(monkeypatch):
    """Qwen3's audio encoder has no cast of its own, so float32 features crash a bf16 model."""
    backend = attached(monkeypatch)
    encoded, _ = backend._encode(ModelInput("q"))
    assert encoded["cast_to"] == torch.bfloat16
