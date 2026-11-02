"""ffmpeg for machines without it on PATH: falls back to the binary bundled with imageio-ffmpeg. That
bundle has no ffprobe, so a duration probe falls back to PyAV."""
from __future__ import annotations

import shutil
import subprocess


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise FileNotFoundError("no ffmpeg on PATH and imageio-ffmpeg is not installed") from e
    return imageio_ffmpeg.get_ffmpeg_exe()


def media_seconds(path: str) -> float:
    import av

    with av.open(path) as container:
        return container.duration / av.time_base


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run for commands written as ["ffmpeg", ...] or ["ffprobe", ...]."""
    if cmd[0] == "ffmpeg":
        cmd = [ffmpeg_exe(), *cmd[1:]]
    elif cmd[0] == "ffprobe" and not shutil.which("ffprobe"):
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{media_seconds(cmd[-1])}\n", stderr="")
    return subprocess.run(cmd, **kwargs)
