"""Qwen3-Omni and Qwen2.5-Omni thinkers. Qwen3 is tested on the real class with a tiny config, Qwen2.5
is untested; neither has run on real weights."""
from __future__ import annotations

import torch

from .hf_common import HFBackend, media_kwargs, model_dtype, read_video, sampled_video_kwargs


class Qwen3OmniBackend(HFBackend):
    """Qwen3OmniMoeThinkerForConditionalGeneration gives text-only output."""

    def _load(self):
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        processor = Qwen3OmniMoeProcessor.from_pretrained(self.config.get("processor_id") or self.config["model_id"])
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            self.config["model_id"], dtype=model_dtype(self.config), **self._placement())
        return processor, model

    def _placement(self) -> dict:
        """One GPU, or the rank's GPUs when weights outgrow one: 63.4 GB at bf16."""
        gpus = self.config.get("gpus")
        if not gpus:
            return {"device_map": self.device}
        reserve = self.config.get("gpu_reserve_gib", 8) * 2**30
        return {"device_map": "balanced", "max_memory": {g: torch.cuda.get_device_properties(g).total_memory - reserve for g in gpus}}

    def _decoder(self):
        return self._base().model

    def _media(self, inputs):
        settings = media_kwargs(self.config)
        num_frames = settings.pop("num_frames", None)
        if num_frames is None or inputs.video is None:
            return settings, str
        frames, metadata = read_video(inputs.video)
        return {**settings, **sampled_video_kwargs(num_frames, metadata)}, lambda _: frames

    def _generate_kwargs(self) -> dict:
        """The thinker's own configs carry no EOS (the composite class passes it to the thinker), so
        without this generation runs to max_new_tokens and every path breaks ValidFormat."""
        tokenizer = self.processor.tokenizer
        return {"eos_token_id": tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]), "pad_token_id": tokenizer.pad_token_id}

    def _logits(self, encoded, flag, start):
        """The head runs on the completion only: the prompt's ~2k video tokens cost ~1 GB of logits each."""
        handle = self._base().lm_head.register_forward_pre_hook(lambda _, args: (args[0][:, start:],))
        try:
            return self.model(**encoded, **{self.flag: flag}, use_cache=False).logits
        finally:
            handle.remove()


class Qwen2_5OmniBackend(HFBackend):
    def _load(self):
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        processor = Qwen2_5OmniProcessor.from_pretrained(self.config["model_id"])
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.config["model_id"], dtype=model_dtype(self.config), device_map=self.device)
        return processor, model
