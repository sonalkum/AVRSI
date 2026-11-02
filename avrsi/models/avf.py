"""Audio-Visual Flamingo. UNTESTED, and its API is unseen: it needs the lashahub/transformers fork,
branch add_AudioVisualFlamingo, not in this env. Auto classes and processor kwargs are assumptions to
confirm in Phase 0, with the §4 fallback if it cannot backprop. Research use only (T12)."""
from __future__ import annotations

from .hf_common import HFBackend, model_dtype


class AVFBackend(HFBackend):
    def _load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.config["model_id"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"], torch_dtype=model_dtype(self.config), device_map=self.device, trust_remote_code=True)
        return processor, model
