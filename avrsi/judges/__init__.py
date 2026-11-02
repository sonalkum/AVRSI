from .base import Judge, JudgeError, Media, SpendCapExceeded, load_key
from .gemini import GeminiJudge
from .openai import OpenAIJudge


def make_judge(config: dict, scope: str | None = None, **kwargs) -> Judge:
    """scope names a spend cap under config["caps"], with its own ledger: an arm, gate0 or audit."""
    if scope is not None:
        config = {**config, **config["caps"][scope]}
    return {"gemini": GeminiJudge, "openai": OpenAIJudge}[config["provider"]](config, scope=scope, **kwargs)


__all__ = ["GeminiJudge", "Judge", "JudgeError", "Media", "OpenAIJudge", "SpendCapExceeded", "load_key", "make_judge"]
