"""Benchmark adapters. AVHBench and CMM have confirmed schemas, AVQA a documented one with its guesses
marked in avqa.py; the rest raise until Phase 0."""
from __future__ import annotations

from functools import partial
from typing import Callable

from ..parse import Item
from .avhbench import load_avhbench
from .avqa import load_avqa
from .cmm import load_cmm


def _pending(name: str, verify: str) -> Callable[..., list[Item]]:
    def load(*_args, **_kwargs):
        raise NotImplementedError(f"{name}: no adapter yet. [VERIFY] {verify} (PLAN §5, Phase 0)")
    return load


REGISTRY: dict[str, Callable[..., list[Item]]] = {
    "avhbench": load_avhbench,
    "cmm": load_cmm,
    "avqa": load_avqa,
    "av_confusebench": _pending("AV-ConfuseBench", "download, licence and schema; videos are separate"),
    "svhalluc": _pending("SVHalluc", "data release"),
    "c3po": _pending("C3PO", "release"),
    "mvhbench": _pending("MVHBench", "release"),
    "worldsense": _pending("WorldSense", "download, licence and schema"),
    "dailyomni": _pending("DailyOmni", "download, licence and schema"),
    "omnibench": _pending("OmniBench", "download, licence and schema"),
    "video_mme": _pending("Video-MME", "download, licence and schema"),
    "avqa_test": partial(load_avqa, split="val"),
    "valor2": _pending("Valor32k-AVQA v2.0", "download, licence and schema"),
    "music_avqa": _pending("MUSIC-AVQA", "download, licence and schema"),
    "mmou": _pending("MMOU", "download, licence and schema"),
    "mmau_test_mini": _pending("MMAU test-mini", "download, licence and schema"),
}


def load_benchmark(name: str, *args, **kwargs) -> list[Item]:
    if name not in REGISTRY:
        raise KeyError(f"unknown benchmark {name}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](*args, **kwargs)
