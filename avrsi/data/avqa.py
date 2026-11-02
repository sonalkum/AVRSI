"""AVQA (Yang et al. 2022), schema as documented at mn.cs.tsinghua.edu.cn/avqa: train_qa.json and
val_qa.json with rows {id, video_name, video_id, question_text, multi_choice, answer, question_relation,
question_type}; the page shows one row.

Guesses, none checked against the files: [VERIFY] each file is a JSON list of such rows; [VERIFY] `answer`
is a 0-based index into multi_choice (the page's example has 0 and says "index"); [VERIFY] clips are
<video_root>/<video_name>.mp4, which depends on how they are cut from YouTube with the page's CSV.
The official download has no clips with it. A row whose clip is missing is dropped and counted."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..parse import Item

RELATIONS = ("View", "Sound", "Both")
FILES = {"train": "train_qa.json", "val": "val_qa.json"}


def load_avqa(root: str | Path, split: str = "train", video_root: str | Path | None = None, check_media: bool = True) -> list[Item]:
    root = Path(root)
    videos = Path(video_root) if video_root else root / "videos"
    present = set(os.listdir(videos)) if check_media else None
    rows = json.loads((root / FILES[split]).read_text())
    items = []
    for row in rows:
        options = tuple(row["multi_choice"])
        if not 0 <= row["answer"] < len(options):
            raise ValueError(f"AVQA {row['id']}: answer {row['answer']} outside {len(options)} choices")
        if row["question_relation"] not in RELATIONS:
            raise ValueError(f"AVQA {row['id']}: unknown question_relation {row['question_relation']!r}")
        name = f"{row['video_name']}.mp4"
        if present is not None and name not in present:
            continue
        items.append(Item(f"avqa-{row['id']}", row["question_text"], options[row["answer"]], options, video=str(videos / name),
                          meta={"video_name": row["video_name"], "video_id": row["video_id"], "relation": row["question_relation"],
                                "question_type": row["question_type"]}))
    if len(items) < len(rows):
        print(f"AVQA {split}: dropped {len(rows) - len(items)} of {len(rows)} rows with no clip")
    return items
