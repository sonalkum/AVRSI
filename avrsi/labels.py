"""SigLIP frame scoring and CLAP/SigLIP class similarity (PLAN §5.2). The scoring rules are tested;
the model calls are UNTESTED on real models, and SigLIP and CLAP checkpoints are config values."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .perturb import Clip

N_FRAMES = 8


def uniform_frame_indices(n_available: int, n: int = N_FRAMES) -> list[int]:
    return np.linspace(0, n_available - 1, n).round().astype(int).tolist()


def max_frame_scores(frame_scores: np.ndarray) -> np.ndarray:
    """[frames, classes] -> [classes]: §5.2 scores a class as its max over the frames."""
    return np.asarray(frame_scores).max(0)


def score_clip(clip_id: str, source_video: str, label: str, frames: Sequence, classes: Sequence[str],
               frame_scorer: Callable[[Sequence, Sequence[str]], np.ndarray]) -> Clip:
    """A row for perturb.load_clips: the label's score, plus every other class as a partner."""
    scores = dict(zip(classes, max_frame_scores(frame_scorer(frames, classes)).tolist()))
    return Clip(clip_id, source_video, label, scores[label], {c: s for c, s in scores.items() if c != label})


def cosine_similarity(embeddings: Mapping[str, np.ndarray]) -> dict[frozenset, float]:
    names = sorted(embeddings)
    unit = np.stack([np.asarray(embeddings[n], float) / np.linalg.norm(embeddings[n]) for n in names])
    table = unit @ unit.T
    return {frozenset((a, b)): float(table[i, j]) for i, a in enumerate(names) for j, b in enumerate(names) if i < j}


def class_similarity(clap: Mapping[str, np.ndarray], siglip: Mapping[str, np.ndarray]) -> dict[frozenset, float]:
    """A pair is as close as its closer modality: close in either must leave the label-disjoint tier."""
    a, b = cosine_similarity(clap), cosine_similarity(siglip)
    return {pair: max(a[pair], b[pair]) for pair in a}


def write_similarity(table: Mapping[frozenset, float], path: str | Path) -> None:
    """The file build_xm.py reads as --similarity."""
    Path(path).write_text(json.dumps({"|".join(sorted(k)): v for k, v in table.items()}))


class SiglipScorer:
    """UNTESTED on real models. Zero-shot sigmoid scores; call as a frame_scorer."""

    def __init__(self, model_id: str, device: str = "cuda"):
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    def __call__(self, frames: Sequence, classes: Sequence[str]) -> np.ndarray:
        import torch

        texts = [f"a photo of {c}" for c in classes]
        inputs = self.processor(text=texts, images=list(frames), padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            return torch.sigmoid(self.model(**inputs).logits_per_image).float().cpu().numpy()

    def text_embeddings(self, classes: Sequence[str]) -> dict[str, np.ndarray]:
        import torch

        inputs = self.processor(text=[f"a photo of {c}" for c in classes], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs).float().cpu().numpy()
        return dict(zip(classes, features))


def clap_text_embeddings(model_id: str, classes: Sequence[str], device: str = "cuda") -> dict[str, np.ndarray]:
    """UNTESTED on real models."""
    import torch
    from transformers import AutoProcessor, ClapModel

    processor = AutoProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).to(device).eval()
    inputs = processor(text=[f"the sound of {c}" for c in classes], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs).float().cpu().numpy()
    return dict(zip(classes, features))
