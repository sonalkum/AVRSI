"""CMM: all_data_final_reorg.json from HF DAMO-NLP-SG/CMM, keys as in that file.

[VERIFY] sample_id. PLAN clusters by sample, but all 2,400 rows have distinct media paths, so paths do
not reveal the 1,200 samples. Rows alternate yes/no in pairs (1,198 of 1,200 consecutive pairs), so
sample_id is the consecutive pair index. Confirm against media content hashes in Phase 0 and pass
sample_ids to override."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..parse import Item

SUBCATEGORIES = {
    "overrely_language_ignore_visual": "Language Dominance",
    "overrely_visual_ignore_audio": "Visual Dominance",
    "overrely_audio_ignore_visual": "Audio Dominance",
    "visual-language": "Visual-Language",
    "audio-language": "Audio-Language",
    "visual-audio-language": "Visual-Audio-Language",
}


def load_cmm(annotations: str | Path, root: str | Path = "", sample_ids: Sequence | None = None) -> list[Item]:
    root = Path(root)
    items = []
    for i, row in enumerate(json.loads(Path(annotations).read_text())):
        if row["sub_category"] not in SUBCATEGORIES:
            raise ValueError(f"unknown CMM sub_category {row['sub_category']!r}")
        answer = row["answer"].strip().lower()
        if answer not in ("yes", "no"):
            raise ValueError(f"row {i}: expected yes/no, got {row['answer']!r}")
        video = str(root / row["video_path"]) if row["video_path"] else None
        audio = str(root / row["audio_path"]) if row["audio_path"] else None
        modalities = tuple(m for m, src in (("audio", audio), ("video", video)) if src)
        sample = sample_ids[i] if sample_ids is not None else i // 2
        items.append(Item(f"cmm-{i:06d}", row["question"], answer, video=video, audio=audio, modalities=modalities,
                          meta={"subcategory": SUBCATEGORIES[row["sub_category"]], "sample_id": sample}))
    return items
