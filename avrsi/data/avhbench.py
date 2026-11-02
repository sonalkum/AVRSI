"""AVHBench public subset: JSON list of {video_id, task, text, label} (github.com/kaist-ami/AVHBench).
Task strings are as in that README's example. Which of the file's rows are the swapped samples is not
documented."""
from __future__ import annotations

import json
from pathlib import Path

from ..parse import Item

TASKS = {
    "Audio-driven Video Hallucination": "a2v",
    "Video-driven Audio Hallucination": "v2a",
    "AV Matching": "matching",
    "AV Captioning": "captioning",
}


def load_avhbench(qa_json: str | Path, video_root: str | Path = "", tasks: tuple[str, ...] | None = None) -> list[Item]:
    items = []
    for i, row in enumerate(json.loads(Path(qa_json).read_text())):
        if row["task"] not in TASKS:
            raise ValueError(f"unknown AVHBench task {row['task']!r}")
        task = TASKS[row["task"]]
        if tasks is not None and task not in tasks:
            continue
        label = row["label"]
        if task != "captioning":
            label = label.strip().lower()
            if label not in ("yes", "no"):
                raise ValueError(f"row {i}: expected Yes/No label, got {row['label']!r}")
        video = str(Path(video_root) / f"{row['video_id']}.mp4")
        items.append(Item(f"avhbench-{i:06d}", row["text"], label, video=video,
                          meta={"task": task, "video_id": row["video_id"]}))
    return items
