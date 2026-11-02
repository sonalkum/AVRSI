"""Shared Transformers plumbing for the real wrappers. Exercised on the real Qwen3-Omni thinker class
with a tiny random config and real PEFT (tests/test_qwen3_real.py); never yet on real weights or a GPU."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .base import ACTOR, REFERENCE, Backend, Generation, ModelInput, Sampling, WithholdingError, assert_withheld

AUDIO_RATE = 16000
FILL_AUDIO_SECONDS, FILL_FRAMES, FILL_SIZE = 10, 16, 224
FRAME_EPS = 1e-3  # sample_frames floors its frame count; this keeps N from rounding to N - 2


@lru_cache(maxsize=8)  # the passes of one question reuse its clip
def load_audio(path: str) -> np.ndarray:
    """Mono float32 at 16 kHz from any container, with no ffmpeg binary. librosa cannot read mp4 without one."""
    import av

    with av.open(path) as container:
        if not container.streams.audio:
            raise WithholdingError(f"{path}: no audio stream")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_RATE)
        chunks = [out.to_ndarray()[0] for frame in container.decode(audio=0) for out in resampler.resample(frame)]
        chunks += [out.to_ndarray()[0] for out in resampler.resample(None)]
    return np.concatenate(chunks)


@lru_cache(maxsize=8)
def read_video(path: str):
    """All frames (T, H, W, 3) and their metadata through PyAV. Transformers' default decoders here
    are torchcodec, broken without FFmpeg libraries, and torchvision, deprecated."""
    import av
    from transformers.video_utils import VideoMetadata

    with av.open(path) as container:
        stream = container.streams.video[0]
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(stream)])
        fps = float(stream.average_rate)
    return frames, VideoMetadata(total_num_frames=len(frames), fps=fps, duration=len(frames) / fps, height=frames.shape[1],
                                 width=frames.shape[2], video_backend="pyav")


def sampled_video_kwargs(num_frames: int, metadata) -> dict:
    """T7: num_frames reaches the model only with do_sample_frames=True, and sample_frames refuses
    num_frames and fps together. The processor also reads fps to place video against audio, defaulting
    to 1. So N is asked for as the clip's own rate N / duration, which is consistent for both."""
    frames = min(num_frames, metadata.total_num_frames)
    return {"do_sample_frames": True, "fps": (frames + FRAME_EPS) / metadata.duration, "video_metadata": metadata}


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
    """One base model, PEFT adapters "actor" and "critic"; θ_0 is the base with adapters off."""

    _lora = False

    def __init__(self, config: dict, device: str = "cuda"):
        self.config, self.device = config, device
        self.flag = config["audio_in_video_flag"]
        self.logprob_batch = config.get("logprob_batch") or 1
        self.generate_batch = config.get("generate_batch") or 1
        self.processor, self.model = self._load()

    def _load(self):
        raise NotImplementedError

    def _base(self):
        return self.model.get_base_model() if self._lora else self.model

    def _decoder(self):
        """The module whose layers checkpoint; None disables gradient checkpointing."""
        return None

    def _media(self, inputs: ModelInput) -> tuple[dict, object]:
        """Processor settings for this input and how a path becomes the processor's video."""
        return media_kwargs(self.config), str

    def _generate_kwargs(self) -> dict:
        return {}

    def _logits(self, encoded, flag: bool, start: int) -> torch.Tensor:
        """Logits from position `start` on."""
        return self.model(**encoded, **{self.flag: flag}, use_cache=False).logits[:, start:]

    def attach_lora(self, r: int) -> None:
        targets = self.config["lora_target_modules"]
        if not targets:
            raise ValueError("lora_target_modules unset: [VERIFY] Phase 0")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("peft is required for training: see requirements-qwen3.txt") from e
        if self.config.get("gradient_checkpointing") and self._decoder() is not None:
            # Before PEFT wraps it. Non-reentrant: the backward pass re-runs layers, whose inputs carry no grad.
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = LoraConfig(r=r, lora_alpha=self.config.get("lora_alpha", 8), target_modules=targets,
                          exclude_modules=self.config.get("lora_exclude_modules"), init_lora_weights=True)
        self.model = get_peft_model(self.model, lora, adapter_name="actor")
        self.model.add_adapter("critic", lora)
        self.sync_critic()
        self.model.set_adapter("actor")
        self.model.eval()
        self._lora = True

    def save_adapters(self, path) -> None:
        """Non-default adapters land in <path>/<adapter_name>/."""
        self.model.save_pretrained(str(path), selected_adapters=["actor", "critic"], save_embedding_layers=False)

    def load_adapters(self, path, trainable: bool = True) -> None:
        """Replaces the weights of the attached adapters in place."""
        self.model.load_adapter(str(Path(path) / "actor"), adapter_name="actor", is_trainable=trainable)
        self.model.load_adapter(str(Path(path) / "critic"), adapter_name="critic", is_trainable=False)
        self.model.set_adapter("actor")

    @contextmanager
    def _role(self, role: str):
        """Always ends with the actor active: set_adapter also moves requires_grad, and the checkpointed
        backward pass re-runs layers under whatever adapter is then active."""
        if not self._lora:
            yield
            return
        try:
            if role == REFERENCE:
                with self.model.disable_adapter():
                    yield
            else:
                self.model.set_adapter("actor" if role == ACTOR else "critic")
                yield
        finally:
            self.model.set_adapter("actor")

    @contextmanager
    def _checkpointed(self):
        """Layers checkpoint only in training mode, and only the decoder needs it; the towers stay in eval."""
        decoder = self._decoder()
        if decoder is None or not self.config.get("gradient_checkpointing"):
            yield
            return
        decoder.train()
        try:
            yield
        finally:
            decoder.eval()

    def _encode(self, inputs: ModelInput):
        text = self.processor.apply_chat_template(messages(inputs), add_generation_prompt=True, tokenize=False)
        settings, video_loader = self._media(inputs)
        kwargs = processor_kwargs(inputs, text, self.flag, video_loader=video_loader, settings=settings)
        flag = kwargs[self.flag]
        encoded = self.processor(**kwargs)
        assert_withheld(inputs, encoded)
        seconds = encoded.get("video_second_per_grid")  # stays float: bf16 would round the position scale
        encoded = encoded.to(self.device).to(model_dtype(self.config))
        if seconds is not None:
            encoded["video_second_per_grid"] = seconds.to(self.device)
        return encoded, flag

    @torch.no_grad()
    def generate(self, inputs: Sequence[ModelInput], sampling: Sampling, role: str = ACTOR) -> list[Generation]:
        """Runs of the same input (the K rollouts) share one call, up to generate_batch at a time."""
        outputs = []
        extra = self._generate_kwargs()
        stop = extra.get("eos_token_id", [])
        with self._role(role):
            for x, n in _runs(inputs, self.generate_batch):
                encoded, flag = self._encode(x)
                sample = {"temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k} if sampling.do_sample else {}
                if n > 1 and sampling.do_sample:
                    sample["num_return_sequences"] = n
                ids = self.model.generate(**encoded, **sample, **extra, do_sample=sampling.do_sample, repetition_penalty=sampling.repetition_penalty,
                                          max_new_tokens=sampling.max_new_tokens, **{self.flag: flag})
                rows = ids[:, encoded["input_ids"].shape[1]:].tolist()
                for row in rows if sampling.do_sample else rows * n:  # greedy repeats are the same path
                    new = _through_eos(row, stop)
                    outputs.append(Generation(self.processor.tokenizer.decode(new, skip_special_tokens=True), new))
        return outputs

    def score_logprobs(self, inputs, completions: Sequence[Generation], role: str = ACTOR) -> list[torch.Tensor]:
        """Runs of the same input are scored together, up to logprob_batch at a time, right-padded."""
        scores = []
        grad = torch.is_grad_enabled() and role == ACTOR
        with torch.set_grad_enabled(grad), self._role(role), (self._checkpointed() if grad else nullcontext()):
            start = 0
            for x, n in _runs(inputs, self.logprob_batch):
                encoded, flag = self._encode(x)
                ids = [g.token_ids for g in completions[start:start + n]]
                start += n
                prompt = encoded["input_ids"].shape[1]
                length = max(map(len, ids))
                new = torch.tensor([row + [0] * (length - len(row)) for row in ids], device=self.device)
                mask = torch.tensor([[1] * len(row) + [0] * (length - len(row)) for row in ids], device=self.device)
                batch = {k: _repeat(v, n) for k, v in encoded.items()}
                batch["input_ids"] = torch.cat([batch["input_ids"], new], 1)
                batch["attention_mask"] = torch.cat([batch["attention_mask"], mask], 1)
                logits = self._logits(batch, flag, prompt - 1)[:, :length]
                logps = torch.log_softmax(logits.float(), -1).gather(2, new[..., None].to(logits.device)).squeeze(2)  # bf16 is too coarse for the KL
                scores += [row[:len(g)] for row, g in zip(logps, ids)]
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


def _runs(inputs: Sequence, limit: int):
    """(input, count) for consecutive equal inputs, each count at most limit."""
    for x, group in groupby(inputs):
        total = len(list(group))
        for start in range(0, total, limit):
            yield x, min(limit, total - start)


def _through_eos(ids: list[int], stop: Sequence[int]) -> list[int]:
    """Batched generation pads after EOS; the EOS itself is part of the trajectory."""
    for i, token in enumerate(ids):
        if token in stop:
            return ids[:i + 1]
    return ids


def _repeat(value, n: int):
    """Every processor output concatenates over the batch, patches and grids included, so n copies of a
    sample are a batch of n."""
    return value.repeat(n, *[1] * (value.dim() - 1)) if n > 1 and isinstance(value, torch.Tensor) else value
