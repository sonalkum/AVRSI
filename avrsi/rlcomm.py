"""A7 reward terms, inspired by RL-CoMM eqs 2-4 (PLAN appendix). The LALM reference and the coherence
scorer are injected; load_rl_comm_scorer's real models are UNTESTED."""
from __future__ import annotations

from typing import Callable, Sequence

from .parse import Item, ParsedPath, canonical_answer, gold_answer

OMEGA = 0.8
S_TASK = "Judge whether the given query is semantically consistent with the provided content"
I_TASK = "Given a query, retrieve semantically coherent content"
Coherence = Callable[[str, str, str], float]  # (task, query, content) -> score


def r_arr(audio: str, reference: str, correct: bool, coherence: Coherence) -> float:
    """Eq 3."""
    return float(correct and coherence(S_TASK, audio, reference) > OMEGA)


def r_avc(audio: str, visual: str, answered: bool, correct: bool, coherence: Coherence) -> float:
    """Eq 4."""
    return float(correct) + coherence(I_TASK, audio, visual) if answered else 0.0


class RLCommScorer:
    """(r_arr, r_avc) per path. reference(item) is the LALM's audio reasoning, one per item."""

    def __init__(self, reference: Callable[[Item], str], coherence: Coherence):
        self.reference, self.coherence = reference, coherence
        self.references: dict[str, str] = {}

    def __call__(self, item: Item, paths: Sequence[ParsedPath]) -> list[tuple[float, float]]:
        if item.item_id not in self.references:
            self.references[item.item_id] = self.reference(item)
        result = []
        for path in paths:
            answered = path.valid and canonical_answer(path.answer, item) is not None
            correct = answered and canonical_answer(path.answer, item) == gold_answer(item)
            audio, visual = " ".join(path.audio), " ".join(path.visual)
            result.append((r_arr(audio, self.references[item.item_id], correct, self.coherence),
                           r_avc(audio, visual, answered, correct, self.coherence)))
        return result


LALM_PROMPT = "Question: {question}\nThe correct answer is: {gold}\nDescribe what is heard in the audio that leads to this answer."


def embedding_coherence(model_id: str, device: str = "cuda") -> Coherence:
    """UNTESTED on real models. Cosine with Qwen3-Embedding's instruct prompt and last-token pooling [VERIFY]."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    def score(task: str, query: str, content: str) -> float:
        batch = tokenizer([f"Instruct: {task}\nQuery:{query}", content], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embedded = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, -1], dim=1)
        return float(embedded[0] @ embedded[1])

    return score


def lalm_reference(model_id: str, device: str = "cuda", max_new_tokens: int = 256) -> Callable[[Item], str]:
    """UNTESTED on real models. Audio Flamingo 3; RL-CoMM's reference sees the ground truth. Class and message shape [VERIFY]."""
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id).to(device).eval()

    def reference(item: Item) -> str:
        text = {"type": "text", "text": LALM_PROMPT.format(question=item.question, gold=gold_answer(item))}
        conversation = [{"role": "user", "content": [text, {"type": "audio", "path": item.audio or item.video}]}]
        inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True).to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        return processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    return reference


def load_rl_comm_scorer(config: dict, device: str = "cuda") -> RLCommScorer:
    return RLCommScorer(lalm_reference(config["lalm"], device), embedding_coherence(config["embedding"], device))
