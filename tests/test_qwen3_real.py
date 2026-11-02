"""HFBackend on the real thinker class with real PEFT: a tiny random config on CPU. The multimodal path
through the real processor is in test_qwen3_preprocess.py."""
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")
from peft.tuners.lora import LoraLayer  # noqa: E402
from qwen3_tiny import BASE, LAYERS, tiny_config, tiny_thinker  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import BatchFeature, Qwen3OmniMoeThinkerForConditionalGeneration as Thinker  # noqa: E402

from avrsi.models.base import ACTOR, CRITIC, REFERENCE, Generation, ModelInput, Sampling  # noqa: E402
from avrsi.models.hf_common import _through_eos  # noqa: E402
from avrsi.models.qwen import Qwen3OmniBackend  # noqa: E402

R = 4
IM_END = 151645
PROMPT = torch.tensor([[872, 300, 301, 302, 303, 304, 305, 306]])
GREEDY = Sampling(0.0, 1.0, False, 6)
SAMPLED = Sampling(1.0, 1.0, True, 6)
X = ModelInput("q")
COMPLETION = Generation("", [11, 12, 13, 14])


class Text:
    """Stands in for the tokenizer and processor: text only."""
    pad_token_id = 151643

    def __init__(self):
        self.tokenizer = self

    def convert_tokens_to_ids(self, tokens):
        return [{"<|im_end|>": IM_END, "<|endoftext|>": self.pad_token_id}[t] for t in tokens]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))

    def encode(self, text, add_special_tokens=False):
        return [1]

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        return "text"

    def __call__(self, **kwargs):
        return BatchFeature({"input_ids": PROMPT.clone(), "attention_mask": torch.ones_like(PROMPT)})


class Tiny(Qwen3OmniBackend):
    def __init__(self, config, model):
        self._model = model
        super().__init__(config, device="cpu")

    def _load(self):
        return Text(), self._model


def build(lora=True, seed=0, **settings):
    backend = Tiny({**BASE, "lora_exclude_modules": r"(audio_tower|visual)\..*", **settings}, tiny_thinker(seed))
    if lora:
        torch.manual_seed(seed)
        backend.attach_lora(R)
    return backend


@pytest.fixture(scope="module")
def shared():
    return build(gradient_checkpointing=True)


@pytest.fixture
def backend():
    return build()


def bump(params, by=0.1):
    for p in params:
        p.data.add_(by)


def named(backend, adapter):
    return {n: p for n, p in backend.model.named_parameters() if f".{adapter}." in n and "lora_" in n}


def base_logits(backend, ids):
    with backend.model.disable_adapter(), torch.no_grad():
        return backend.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits


def plain_thinker_logits(ids):
    with torch.no_grad():
        return tiny_thinker(0)(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits


def test_adapters_cover_decoder_attention_only(backend):
    layers = {n for n, m in backend.model.named_modules() if isinstance(m, LoraLayer)}
    assert len(layers) == 4 * LAYERS and all(".model.layers." in n and ".self_attn." in n for n in layers)
    assert set(backend.model.peft_config) == {"actor", "critic"}


def test_gradients_reach_only_actor_lora(backend):
    logps = backend.score_logprobs([X], [COMPLETION], ACTOR)[0]
    logps.sum().backward()
    grads = {n: p.grad for n, p in backend.model.named_parameters() if p.grad is not None}
    assert set(grads) == set(named(backend, "actor"))
    assert any(g.abs().sum() > 0 for n, g in grads.items() if "lora_B" in n)
    assert not any(p.requires_grad for n, p in backend.model.named_parameters() if n not in grads)


def test_generate_differs_by_role_and_reference_is_the_base(backend):
    ids = PROMPT
    assert torch.equal(base_logits(backend, ids), plain_thinker_logits(ids).to(base_logits(backend, ids).dtype))
    bump(backend.trainable_parameters())
    for role in (ACTOR, CRITIC, REFERENCE):
        assert len(backend.generate([X], GREEDY, role)[0].token_ids) == GREEDY.max_new_tokens
    actor = backend.score_logprobs([X], [COMPLETION], ACTOR)[0]
    critic = backend.score_logprobs([X], [COMPLETION], CRITIC)[0]
    reference = backend.score_logprobs([X], [COMPLETION], REFERENCE)[0]
    assert torch.allclose(critic, reference)  # θ_c is still the zero-init copy, so it equals θ_0
    assert not torch.allclose(actor, reference)
    assert backend.model.active_adapters == ["actor"] and all(p.requires_grad for p in backend.trainable_parameters())


def test_reference_matches_a_bare_thinker(backend):
    bump(backend.trainable_parameters())
    new = torch.tensor([COMPLETION.token_ids])
    logits = plain_thinker_logits(torch.cat([PROMPT, new], 1))[0, PROMPT.shape[1] - 1:-1].float()
    expected = torch.log_softmax(logits, -1).gather(1, new.T).squeeze(1)
    assert torch.allclose(backend.score_logprobs([X], [COMPLETION], REFERENCE)[0], expected)


def test_sync_copies_actor_to_critic_without_alias(backend):
    bump(backend.trainable_parameters())
    backend.sync_critic()
    actor, critic = named(backend, "actor"), named(backend, "critic")
    assert len(actor) == len(critic) == 8 * LAYERS
    for name, param in actor.items():
        assert torch.equal(param, critic[name.replace(".actor.", ".critic.")])
    bump(backend.trainable_parameters())
    assert not any(torch.equal(p, critic[n.replace(".actor.", ".critic.")]) for n, p in actor.items())


def test_exception_inside_a_role_restores_the_actor(backend):
    for role in (CRITIC, REFERENCE):
        with pytest.raises(RuntimeError):
            with backend._role(role):
                raise RuntimeError
        assert backend.model.active_adapters == ["actor"] and all(p.requires_grad for p in backend.trainable_parameters())


def test_adapters_save_and_load(backend, tmp_path):
    bump(backend.trainable_parameters())
    backend.sync_critic()
    bump(backend.trainable_parameters(), 0.2)
    backend.save_adapters(tmp_path)
    assert (tmp_path / "actor" / "adapter_model.safetensors").exists() and (tmp_path / "critic" / "adapter_model.safetensors").exists()
    assert json.loads((tmp_path / "actor" / "adapter_config.json").read_text())["exclude_modules"]
    other = build(seed=1)
    other.load_adapters(tmp_path)
    for adapter in ("actor", "critic"):
        for name, param in named(backend, adapter).items():
            assert torch.equal(param, named(other, adapter)[name])
    assert other.model.active_adapters == ["actor"]
    assert all(p.requires_grad for p in other.trainable_parameters())
    assert not any(p.requires_grad for p in named(other, "critic").values())
    assert {p.dtype for p in other.trainable_parameters()} == {torch.float32}
    assert len({n for n, m in other.model.named_modules() if isinstance(m, LoraLayer)}) == 4 * LAYERS
    optimizer = torch.optim.AdamW(other.trainable_parameters(), lr=1e-3)
    other.score_logprobs([X], [COMPLETION], ACTOR)[0].sum().backward()
    before = other.trainable_parameters()[1].detach().clone()
    optimizer.step()
    assert not torch.equal(before, other.trainable_parameters()[1])


def test_logprobs_match_a_manual_forward(backend):
    bump(backend.trainable_parameters())
    new = torch.tensor([COMPLETION.token_ids])
    ids = torch.cat([PROMPT, new], 1)
    with torch.no_grad():
        logits = backend.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, PROMPT.shape[1] - 1:-1].float()
    manual = torch.log_softmax(logits, -1).gather(1, new.T).squeeze(1)
    assert torch.allclose(backend.score_logprobs([X], [COMPLETION], ACTOR)[0].detach(), manual)


def test_logprobs_match_what_generate_sampled_from(backend):
    """Greedy scores from the cached generation path against the uncached forward."""
    bump(backend.trainable_parameters())
    with torch.no_grad():
        out = backend.model.generate(input_ids=PROMPT, attention_mask=torch.ones_like(PROMPT), do_sample=False, max_new_tokens=4,
                                     return_dict_in_generate=True, output_scores=True)
    new = out.sequences[0, PROMPT.shape[1]:].tolist()
    stepped = torch.stack([torch.log_softmax(s[0].float(), -1)[t] for s, t in zip(out.scores, new)])
    assert torch.allclose(backend.score_logprobs([X], [Generation("", new)], ACTOR)[0].detach(), stepped, atol=1e-2)


def test_dtypes_keep_small_steps(backend):
    """bf16 has 8 mantissa bits: a 1e-5 step on a LoRA weight would round to nothing."""
    assert {p.dtype for n, p in backend.model.named_parameters() if "lora_" not in n} == {torch.bfloat16}
    assert {p.dtype for n, p in backend.model.named_parameters() if "lora_" in n} == {torch.float32}
    assert backend.score_logprobs([X], [COMPLETION], ACTOR)[0].dtype == torch.float32
    ids = PROMPT
    assert backend.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.dtype == torch.bfloat16
    optimizer = torch.optim.AdamW(backend.trainable_parameters(), lr=1e-5)
    backend.score_logprobs([X], [COMPLETION], ACTOR)[0].sum().backward()
    before = [p.detach().clone() for p in backend.trainable_parameters()]
    optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, backend.trainable_parameters()))


def actor_grads(backend):
    backend.model.zero_grad()
    backend.score_logprobs([X], [COMPLETION], ACTOR)[0].sum().backward()
    return [p.grad.clone() for p in backend.trainable_parameters()]


def test_checkpointed_gradients_equal_plain_ones():
    plain, checkpointed = build(), build(gradient_checkpointing=True)
    for backend in (plain, checkpointed):
        bump(backend.trainable_parameters())
    assert checkpointed._base().model.layers[0].gradient_checkpointing and not plain._base().model.layers[0].gradient_checkpointing
    for a, b in zip(actor_grads(plain), actor_grads(checkpointed)):
        assert torch.allclose(a, b, atol=1e-5)


def test_checkpointing_actually_runs_and_leaves_eval_mode(shared, monkeypatch):
    calls = []
    layer = shared._base().model.layers[0]
    original = layer._gradient_checkpointing_func
    monkeypatch.setattr(layer, "_gradient_checkpointing_func", lambda *a, **k: calls.append(1) or original(*a, **k))
    shared.score_logprobs([X], [COMPLETION], ACTOR)[0].sum().backward()
    assert calls and not shared.model.training and not shared._base().model.training and not shared._base().audio_tower.training


def test_no_graph_outside_the_actor(shared):
    for role in (CRITIC, REFERENCE):
        assert not shared.score_logprobs([X], [COMPLETION], role)[0].requires_grad
    with torch.no_grad():
        assert not shared.score_logprobs([X], [COMPLETION], ACTOR)[0].requires_grad


def test_batched_scoring_equals_single():
    single, batched = build(), build(logprob_batch=2)
    for backend in (single, batched):
        bump(backend.trainable_parameters())
    gens = [Generation("", [5, 6, 7]), Generation("", [8, 9]), Generation("", [10, 11, 12, 13])]
    expected = single.score_logprobs([X] * 3, gens, ACTOR)
    got = batched.score_logprobs([X] * 3, gens, ACTOR)
    assert [len(g) for g in got] == [3, 2, 4]
    for a, b in zip(expected, got):
        assert torch.allclose(a, b, atol=1e-4)


def test_k_rollouts_share_a_call_and_differ(monkeypatch):
    backend = build(generate_batch=4)
    calls = []
    original = backend.model.generate
    monkeypatch.setattr(backend.model, "generate", lambda **k: calls.append(k) or original(**k))
    gens = backend.generate([X] * 4, SAMPLED)
    assert len(gens) == 4 and len(calls) == 1 and calls[0]["num_return_sequences"] == 4
    assert len({tuple(g.token_ids) for g in gens}) > 1


def test_generation_stops_at_eos_and_keeps_it(monkeypatch):
    backend = build()
    assert backend._generate_kwargs()["eos_token_id"] == [IM_END, Text.pad_token_id]
    out = torch.cat([PROMPT, torch.tensor([[7, IM_END, 151643, 151643]])], 1)
    monkeypatch.setattr(backend.model, "generate", lambda **k: out)
    assert backend.generate([X], GREEDY)[0].token_ids == [7, IM_END]
    assert _through_eos([1, 2], [IM_END]) == [1, 2]


def test_real_generation_stops_at_a_forced_eos():
    backend = build()
    with torch.no_grad():
        backend._base().lm_head.weight[IM_END] += 1e3  # logits favour EOS from the first step
    new = backend.generate([X], Sampling(0.0, 1.0, False, 8))[0].token_ids
    assert new[-1] == IM_END and len(new) < 8


def composite_dir(path):
    """As the hub has it: a thinker under "thinker." in a composite config."""
    model = tiny_thinker(3)
    config = tiny_config()
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_omni_moe", "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
                                                  "enable_audio_output": False, "thinker_config": config.to_dict()}))
    state = {}
    for key, value in model.state_dict().items():
        if key.endswith("experts.gate_up_proj"):  # the hub keeps one Linear per expert
            for i, expert in enumerate(value):
                gate, up = expert.chunk(2, dim=0)
                state[f"thinker.{key[:-12]}{i}.gate_proj.weight"] = gate.clone()
                state[f"thinker.{key[:-12]}{i}.up_proj.weight"] = up.clone()
        elif key.endswith("experts.down_proj"):
            for i, expert in enumerate(value):
                state[f"thinker.{key[:-9]}{i}.down_proj.weight"] = expert.clone()
        else:
            state[f"thinker.{key}"] = value.clone().contiguous()
    save_file(state, path / "model.safetensors")
    return model


def test_from_pretrained_reads_the_thinker_prefix(tmp_path):
    """A wrong prefix or unfused experts would mean silent random weights."""
    original = composite_dir(tmp_path)
    loaded, info = Thinker.from_pretrained(tmp_path, dtype=torch.bfloat16, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    for key, value in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_placement_spans_a_ranks_gpus(monkeypatch):
    backend = SimpleNamespace(config={"gpus": [2, 3], "gpu_reserve_gib": 8}, device="cuda:2")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda g: SimpleNamespace(total_memory=48 * 2**30))
    placed = Qwen3OmniBackend._placement(backend)
    assert placed == {"device_map": "balanced", "max_memory": {2: 40 * 2**30, 3: 40 * 2**30}}
    backend.config = {}
    assert Qwen3OmniBackend._placement(backend) == {"device_map": "cuda:2"}
