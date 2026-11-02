from .base import Judge, JudgeError, Media, SpendCapExceeded, load_key
from .gemini import GeminiJudge
from .openai import OpenAIJudge


def make_judge(config: dict, **kwargs) -> Judge:
    return {"gemini": GeminiJudge, "openai": OpenAIJudge}[config["provider"]](config, **kwargs)


__all__ = ["GeminiJudge", "Judge", "JudgeError", "Media", "OpenAIJudge", "SpendCapExceeded", "load_key", "make_judge"]
