"""Gemini generateContent. Audio and video go inline. Files above the inline limit [VERIFY] Phase 0."""
from __future__ import annotations

from typing import Sequence

from .base import Judge, JudgeError, Media


class GeminiJudge(Judge):
    provider = "gemini"

    def _request(self, prompt: str, media: Sequence[Media]) -> tuple[str, dict, dict]:
        parts = [{"inline_data": {"mime_type": m.mime, "data": m.b64()}} for m in media] + [{"text": prompt}]
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": 0.0}}
        headers = {"x-goog-api-key": self._key.reveal(), "Content-Type": "application/json"}
        return f"{self.base_url}/models/{self.model}:generateContent", headers, body

    def _text(self, response: dict) -> str:
        try:
            return "".join(p.get("text", "") for p in response["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise JudgeError("gemini: response has no text") from None
