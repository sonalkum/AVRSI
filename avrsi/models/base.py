"""Backend interface and modality withholding (PLAN §3.2, §4, T8)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..parse import ALL_MODALITIES, Item

ACTOR, CRITIC, REFERENCE = "actor", "critic", "reference"  # θ, θ_c, θ_0
AUDIO_KEYS = ("input_features", "feature_attention_mask", "audio_features")
VIDEO_KEYS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "video_second_per_grid")


class WithholdingError(RuntimeError):
    """A modality reached the model that should not have, or failed to reach it (T8)."""


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    do_sample: bool
    max_new_tokens: int


@dataclass(frozen=True)
class ModelInput:
    prompt: str
    video: str | None = None
    audio: str | None = None  # file holding the audio stream; may be the video's own container
    fill: str | None = None  # Phase 0 fill standing in for a withheld modality; carries no signal

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(m for m, src in (("audio", self.audio), ("video", self.video)) if src is not None)


@dataclass
class Generation:
    text: str
    token_ids: list[int] = field(default_factory=list)


def build_model_input(item: Item, modalities: Sequence[str], prompt: str, fill: str | None = None) -> ModelInput:
    """Withholding happens here: a removed modality's source never enters ModelInput. Only what the
    sample contains can be included, so CMM unimodal samples stay unimodal."""
    unknown = set(modalities) - set(ALL_MODALITIES)
    if unknown:
        raise ValueError(f"unknown modalities {sorted(unknown)}")
    want = set(modalities) & set(item.modalities)
    video = item.video if "video" in want else None
    audio = (item.audio or item.video) if "audio" in want else None
    if "video" in want and video is None:
        raise WithholdingError(f"{item.item_id}: video requested but the item has none")
    if "audio" in want and audio is None:
        raise WithholdingError(f"{item.item_id}: audio requested but the item has none")
    needs_fill = {"audio": "silent_track", "video": "black_frames"}
    withheld = [m for m in ALL_MODALITIES if m not in want]
    use_fill = fill if fill and len(withheld) == 1 and needs_fill[withheld[0]] == fill else None
    return ModelInput(prompt=prompt, video=video, audio=audio, fill=use_fill)


def assert_withheld(inputs: ModelInput, encoded, tolerate_fill: bool = True) -> None:
    """Checks processor output against the intended modalities, both directions (T8)."""
    keys = {k for k in encoded.keys() if encoded[k] is not None}
    for modality, known in (("audio", AUDIO_KEYS), ("video", VIDEO_KEYS)):
        present = bool(keys & set(known))
        expected = modality in inputs.modalities
        filled = tolerate_fill and inputs.fill == {"audio": "silent_track", "video": "black_frames"}[modality]
        if expected and not present:
            raise WithholdingError(f"{modality} requested but absent from model inputs")
        if not expected and present and not filled:
            raise WithholdingError(f"{modality} withheld but present in model inputs: {sorted(keys & set(known))}")


class Backend(ABC):
    """One model in three roles: ACTOR θ (trainable), CRITIC θ_c, REFERENCE θ_0."""

    supports_logprob_scoring = True

    @abstractmethod
    def generate(self, inputs: Sequence[ModelInput], sampling: Sampling, role: str = ACTOR) -> list[Generation]: ...

    @abstractmethod
    def score_logprobs(
        self, inputs: Sequence[ModelInput], completions: Sequence[Generation], role: str = ACTOR
    ) -> list[torch.Tensor]:
        """Per-token log-probabilities of each completion, differentiable for ACTOR."""

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def sync_critic(self) -> None:
        """θ_c ← θ."""

    @abstractmethod
    def trainable_parameters(self) -> list[torch.nn.Parameter]: ...
