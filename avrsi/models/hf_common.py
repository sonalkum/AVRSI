"""Shared Transformers plumbing for the real wrappers. UNTESTED on real weights: no GPU run has
happened. processor_kwargs and messages are tested on a fake processor; the rest waits for Phase 0."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .base import ACTOR, REFERENCE, Backend, Generation, ModelInput, Sampling, assert_withheld

AUDIO_RATE = 16000
FILL_AUDIO_SECONDS, FILL_FRAMES, FILL_SIZE = 10, 16, 224


def load_audio(path: str) -> np.ndarray:
    import librosa

    return librosa.load(path, sr=AUDIO_RATE, mono=True)[0]


def silent_track(seconds: int = FILL_AUDIO_SECONDS) -> np.ndarray:
    return np.zeros(AUDIO_RATE * seconds, dtype=np.float32)


def black_frames(frames: int = FILL_FRAMES, size: int = FILL_SIZE) -> np.ndarray:
    return np.zeros((frames, size, size, 3), dtype=np.uint8)


def audio_in_video(inputs: ModelInput) -> bool:
    """The audio is the video's own track, so the processor must interleave them. Separate files
    (CMM) stay separate."""
    return inputs.audio is not None and inputs.audio == inputs.video


def media_kwargs(config: dict) -> dict:
    """T7 per-model settings. num_frames reaches the video processor; the AVF kwarg names `size`
    and `audio_window_s` are unseen on the fork: [VERIFY] Phase 0."""
    kwargs = {}
    if config.get("num_frames"):
        kwargs["num_frames"] = config["num_frames"]
    if config.get("image_size"):
        kwargs["size"] = {"height": config["image_size"], "width": config["image_size"]}
    if config.get("audio_window_s"):
        kwargs["audio_window_s"] = config["audio_window_s"]
    return kwargs


def model_dtype(config: dict) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config["dtype"]]


def messages(inputs: ModelInput) -> list[dict]:
    """Media placeholders only for what is sent; a fill has a placeholder and no signal. The Omni
    processors consume one audio length per placeholder, and the video placeholder takes it when
    the flag is set, so a video's own track gets no audio placeholder."""
    content = []
    if (inputs.audio is not None and not audio_in_video(inputs)) or inputs.fill == "silent_track":
        content.append({"type": "audio", "audio": "<audio>"})
    if inputs.video is not None or inputs.fill == "black_frames":
        content.append({"type": "video", "video": "<video>"})
    return [{"role": "user", "content": content + [{"type": "text", "text": inputs.prompt}]}]


def processor_kwargs(inputs: ModelInput, text: str, flag: str, audio_loader=load_audio, video_loader=str,
                     settings: dict | None = None) -> dict:
    """Withheld streams are absent and the audio-in-video flag is true only for a video's own track,
    so that track cannot leak into a V-check."""
    kwargs: dict = {"text": [text], "return_tensors": "pt", "padding": True, **(settings or {})}
    if inputs.audio is not None:
        kwargs["audio"] = [audio_loader(inputs.audio)]
    elif inputs.fill == "silent_track":
        kwargs["audio"] = [silent_track()]
    if inputs.video is not None:
        kwargs["videos"] = [video_loader(inputs.video)]
    elif inputs.fill == "black_frames":
        kwargs["videos"] = [black_frames()]
    kwargs[flag] = audio_in_video(inputs)
    return kwargs


class HFBackend(Backend):
    """UNTESTED. One base model, PEFT adapters "actor" and "critic"; θ_0 is the base with adapters off."""

    _lora = False

    def __init__(self, config: dict, device: str = "cuda"):
        self.config, self.device = config, device
        self.flag = config["audio_in_video_flag"]
        self.processor, self.model = self._load()

    def _load(self):
        raise NotImplementedError

    def attach_lora(self, r: int) -> None:
        targets = self.config["lora_target_modules"]
        if not targets:
            raise ValueError("lora_target_modules unset: [VERIFY] Phase 0")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("peft is required for training and is absent from the gama2 env") from e
        lora = LoraConfig(r=r, target_modules=targets, init_lora_weights=True)
        self.model = get_peft_model(self.model, lora, adapter_name="actor")
        self.model.add_adapter("critic", lora)
        self.sync_critic()
        self.model.set_adapter("actor")
        self._lora = True

    def save_adapters(self, path) -> None:
        self.model.save_pretrained(str(path), selected_adapters=["actor", "critic"])

    def load_adapters(self, path, trainable: bool = True) -> None:
        """[VERIFY] PEFT layout: a non-default adapter saves under <path>/<adapter_name>/, and loading
        over an adapter of the same name replaces its weights."""
        self.model.load_adapter(str(Path(path) / "actor"), adapter_name="actor", is_trainable=trainable)
        self.model.load_adapter(str(Path(path) / "critic"), adapter_name="critic", is_trainable=False)
        self.model.set_adapter("actor")

    def _role(self, role: str):
        if not self._lora:
            return nullcontext()
        if role == REFERENCE:
            return self.model.disable_adapter()
        self.model.set_adapter("actor" if role == ACTOR else "critic")
        return nullcontext()

    def _encode(self, inputs: ModelInput):
        text = self.processor.apply_chat_template(messages(inputs), add_generation_prompt=True, tokenize=False)
        kwargs = processor_kwargs(inputs, text, self.flag, settings=media_kwargs(self.config))
        flag = kwargs[self.flag]
        encoded = self.processor(**kwargs)
        assert_withheld(inputs, encoded)
        return encoded.to(self.device).to(model_dtype(self.config)), flag

    @torch.no_grad()
    def generate(self, inputs: Sequence[ModelInput], sampling: Sampling, role: str = ACTOR) -> list[Generation]:
        outputs = []
        with self._role(role):
            for x in inputs:
                encoded, flag = self._encode(x)
                sample = {"temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k} if sampling.do_sample else {}
                ids = self.model.generate(**encoded, **sample, do_sample=sampling.do_sample, repetition_penalty=sampling.repetition_penalty,
                                          max_new_tokens=sampling.max_new_tokens, **{self.flag: flag})
                new = ids[0, encoded["input_ids"].shape[1]:].tolist()
                outputs.append(Generation(self.processor.tokenizer.decode(new, skip_special_tokens=True), new))
        self._role(ACTOR)  # set_adapter also moves requires_grad, so the actor stays active
        return outputs

    def score_logprobs(self, inputs, completions: Sequence[Generation], role: str = ACTOR) -> list[torch.Tensor]:
        scores = []
        with self._role(role):
            for x, g in zip(inputs, completions):
                encoded, flag = self._encode(x)
                new = torch.tensor([g.token_ids], device=self.device)
                encoded["input_ids"] = torch.cat([encoded["input_ids"], new], 1)
                encoded["attention_mask"] = torch.cat([encoded["attention_mask"], torch.ones_like(new)], 1)
                logits = self.model(**encoded, **{self.flag: flag}).logits[0, -new.shape[1] - 1:-1]
                scores.append(torch.log_softmax(logits, -1).gather(1, new.T).squeeze(1))
        self._role(ACTOR)
        return scores

    def encode(self, text: str) -> list[int]:
        return self.processor.tokenizer.encode(text, add_special_tokens=False)

    def sync_critic(self) -> None:
        params = dict(self.model.named_parameters())
        for name, param in params.items():
            if ".actor." in name and "lora_" in name:
                params[name.replace(".actor.", ".critic.")].data.copy_(param.data)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for n, p in self.model.named_parameters() if "lora_" in n and ".actor." in n]
