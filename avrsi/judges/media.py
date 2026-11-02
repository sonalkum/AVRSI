"""Files for an external judge. Withholding for judges is at file level: a video with its audio
track must not go to a video-only view."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Sequence

from ..parse import Item
from ..perturb import extract_audio_command, strip_audio_command
from .base import Media


def view_media(item: Item, modalities: Sequence[str], work_dir: str | Path, runner: Callable = subprocess.run) -> list[Media]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    want = set(modalities) & set(item.modalities)
    if want == {"audio", "video"}:
        return [Media("video", "video/mp4", item.video)]
    if want == {"video"}:
        out = work / f"{item.item_id}.noaudio.mp4"
        runner(strip_audio_command(item.video, str(out)), check=True)
        return [Media("video", "video/mp4", str(out))]
    if item.audio:
        return [Media("audio", "audio/wav", item.audio)]
    out = work / f"{item.item_id}.wav"
    runner(extract_audio_command(item.video, str(out)), check=True)
    return [Media("audio", "audio/wav", str(out))]
