"""OpenAI chat completions. Video goes as sampled frames plus audio; request shape is [VERIFY API]."""
from __future__ import annotations

from typing import Sequence

from .base import Judge, JudgeError, Media


class OpenAIJudge(Judge):
    provider = "openai"

    def _request(self, prompt: str, media: Sequence[Media]) -> tuple[str, dict, dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for m in media:
            if m.kind == "image":
                content.append({"type": "image_url", "image_url": {"url": f"data:{m.mime};base64,{m.b64()}"}})
            elif m.kind == "audio":
                content.append({"type": "input_audio", "input_audio": {"data": m.b64(), "format": m.mime.split("/")[-1]}})
            else:
                raise JudgeError("openai: pass video as sampled frames plus audio")
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0.0}
        headers = {"Authorization": f"Bearer {self._key.reveal()}", "Content-Type": "application/json"}
        return f"{self.base_url}/chat/completions", headers, body

    def _text(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]
