"""Qwen3-Omni and Qwen2.5-Omni thinkers. UNTESTED on real weights (PLAN §4 facts only)."""
from __future__ import annotations

import torch

from .hf_common import HFBackend


class Qwen3OmniBackend(HFBackend):
    """Qwen3OmniMoeThinkerForConditionalGeneration gives text-only output."""

    def _load(self):
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        processor = Qwen3OmniMoeProcessor.from_pretrained(self.config["model_id"])
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            self.config["model_id"], dtype=torch.bfloat16, device_map=self.device)
        return processor, model


class Qwen2_5OmniBackend(HFBackend):
    def _load(self):
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        processor = Qwen2_5OmniProcessor.from_pretrained(self.config["model_id"])
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.config["model_id"], dtype=torch.bfloat16, device_map=self.device)
        return processor, model
