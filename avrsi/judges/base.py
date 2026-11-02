"""Judge base: key handling, content-hash cache, spend cap (PLAN §6).
Keys come from an env var or a key file, go only in request headers, and are never logged."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import httpx

log = logging.getLogger(__name__)
RETRYABLE = {429, 500, 502, 503, 504}


class JudgeError(RuntimeError):
    pass


class SpendCapExceeded(JudgeError):
    pass


class Secret:
    """Holds a key so that repr, str and format cannot leak it."""

    def __init__(self, value: str):
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__


def load_key(key_env: str | None, key_file: str | None) -> Secret:
    value = os.environ.get(key_env) if key_env else None
    if not value and key_file:
        path = Path(key_file).expanduser()
        if path.exists():
            value = path.read_text().strip()
    if not value:
        raise JudgeError(f"no API key: set {key_env} or provide the key file")
    return Secret(value)


@dataclass(frozen=True)
class Media:
    kind: str  # video, audio, image
    mime: str
    path: str | None = None
    data: bytes | None = None

    def bytes(self) -> bytes:
        return self.data if self.data is not None else Path(self.path).read_bytes()

    def b64(self) -> str:
        return base64.b64encode(self.bytes()).decode()

    def digest(self) -> str:
        return hashlib.sha256(self.bytes()).hexdigest()


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        return json.loads(path.read_text())["text"] if path.exists() else None

    def put(self, key: str, text: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}))


class SpendCap:
    """Calls and estimated USD, persisted so reruns share the cap. usd_per_call None means pricing
    is unresolved [VERIFY] and only max_calls applies."""

    def __init__(self, ledger: str | Path, max_calls: int, max_usd: float, usd_per_call: float | None):
        self.ledger, self.max_calls, self.max_usd, self.usd_per_call = Path(ledger), max_calls, max_usd, usd_per_call
        self.calls = json.loads(self.ledger.read_text())["calls"] if self.ledger.exists() else 0

    @property
    def usd(self) -> float:
        return self.calls * (self.usd_per_call or 0.0)

    def charge(self) -> None:
        over_usd = self.usd_per_call is not None and self.usd + self.usd_per_call > self.max_usd
        if self.calls >= self.max_calls or over_usd:
            raise SpendCapExceeded(f"cap reached: {self.calls} calls, ${self.usd:.2f}")
        self.calls += 1
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.write_text(json.dumps({"calls": self.calls}))


class Judge:
    provider = ""

    def __init__(self, config: dict, transport: httpx.BaseTransport | None = None, cache_dir: str | Path | None = None,
                 sleep: Callable[[float], None] = time.sleep, max_attempts: int = 3):
        self.config, self.sleep, self.max_attempts = config, sleep, max_attempts
        self.model = config["model"]
        self.base_url = config["base_url"].rstrip("/")
        root = Path(cache_dir or config["cache_dir"]) / self.provider
        self.cache = DiskCache(root)
        self.cap = SpendCap(root / "ledger.json", config["max_calls"], config["max_usd"], config["usd_per_call"])
        self._key = load_key(config["key_env"], config["key_file"])
        self._client = httpx.Client(transport=transport, timeout=config["timeout_s"])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def _cache_key(self, prompt: str, media: Sequence[Media]) -> str:
        parts = [self.provider, str(self.model), prompt, *[f"{m.kind}:{m.mime}:{m.digest()}" for m in media]]
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()

    def _request(self, prompt: str, media: Sequence[Media]) -> tuple[str, dict, dict]:
        """Returns url, headers, JSON body."""
        raise NotImplementedError

    def _text(self, response: dict) -> str:
        raise NotImplementedError

    def complete(self, prompt: str, media: Sequence[Media] = ()) -> str:
        if not self.model:
            raise JudgeError(f"{self.provider}: model id unset [VERIFY]")
        key = self._cache_key(prompt, media)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        url, headers, body = self._request(prompt, media)
        for attempt in range(self.max_attempts):
            self.cap.charge()
            try:
                response = self._client.post(url, headers=headers, json=body)
            except httpx.HTTPError as e:
                # httpx messages can quote request detail, so only the type leaves this frame.
                raise JudgeError(f"{self.provider}: transport error {type(e).__name__}") from None
            if response.status_code == 200:
                text = self._text(response.json())
                self.cache.put(key, text)
                log.info("%s %s ok, cache %s", self.provider, self.model, key[:12])
                return text
            log.warning("%s %s HTTP %d", self.provider, self.model, response.status_code)
            if response.status_code not in RETRYABLE or attempt == self.max_attempts - 1:
                raise JudgeError(f"{self.provider}: HTTP {response.status_code}")  # body may echo the request
            self.sleep(2.0 ** attempt)
        raise JudgeError(f"{self.provider}: exhausted attempts")
