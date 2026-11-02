"""Files for an external judge. Withholding for judges is at file level: a video with its audio
track must not go to a video-only view."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .. import ffmpeg
from ..parse import Item
from ..perturb import extract_audio_command, strip_audio_command
from .base import Media


def duration_command(video: str) -> list[str]:
    return ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video]


def frame_command(video: str, at: float, out: str) -> list[str]:
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", video, "-frames:v", "1", out]


def sample_frames(video: str, n: int, work_dir: str | Path, stem: str, runner: Callable = ffmpeg.run) -> list[Media]:
    """n frames at uniform times, each at the middle of its slice of the clip."""
    duration = float(runner(duration_command(video), check=True, capture_output=True, text=True).stdout)
    frames = []
    for i in range(n):
        out = Path(work_dir) / f"{stem}.frame{i}.jpg"
        runner(frame_command(video, duration * (i + 0.5) / n, str(out)), check=True)
        frames.append(Media("image", "image/jpeg", str(out)))
    return frames


def view_media(item: Item, modalities: Sequence[str], work_dir: str | Path, runner: Callable = ffmpeg.run,
               frames: int | None = None) -> list[Media]:
    """frames: send that many sampled frames instead of a video, for judges with no video input."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    want = set(modalities) & set(item.modalities)
    if frames and "video" in want:
        images = sample_frames(item.video, frames, work, item.item_id, runner)
        return images + (view_media(item, ("audio",), work, runner) if "audio" in want else [])
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
