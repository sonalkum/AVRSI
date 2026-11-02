"""Tiny CPU backend for tests: a torch bigram policy over text chunks and a rule-based perceiver.
Media are fake paths such as "fake://dog". A role's leniency scales how often it says supported
regardless of evidence, and travels with θ on sync, so a lenient actor is a lenient judge."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import torch

from .base import ACTOR, CRITIC, REFERENCE, Backend, Generation, ModelInput, Sampling

EOS = "<eos>"


def media_label(path: str | None) -> str | None:
    """"fake://dog" and "fake://root/dog" -> dog; "fake://silence" -> None."""
    if path is None or path.rsplit("/", 1)[-1] == "silence":
        return None
    return path.rsplit("/", 1)[-1].removesuffix(".mp4")


class FakeBackend(Backend):
    supports_logprob_scoring = False

    def __init__(self, classes: Sequence[str] = ("dog", "piano", "rain"), seed: int = 0):
        self.classes = list(classes)
        audio = [f"The sound of {c} can be heard." for c in classes] + ["Something can be heard."]
        visual = [f"The video shows {c}." for c in classes] + ["Something is happening."]
        structure = ["<audio>", "</audio>", "<visual>", "</visual>", "<fusion>", "</fusion>", "<answer>", "</answer>"]
        fusion = ["They agree."]
        answers = ["yes", "no"]
        self.vocab = structure + audio + visual + fusion + answers + [EOS]
        self.ids = {t: i for i, t in enumerate(self.vocab)}
        V = len(self.vocab)
        generator = torch.Generator().manual_seed(seed)
        table = 0.1 * torch.randn(V, V, generator=generator) - 4.0

        def link(sources, targets, weight=4.0):
            for s in sources:
                for t in targets:
                    table[self.ids[s], self.ids[t]] = weight

        link(["<audio>"], audio)
        link(audio, audio, 3.0)
        link(audio, ["</audio>"])
        link(["</audio>"], ["<visual>"])
        link(["<visual>"], visual)
        link(visual, visual, 3.0)
        link(visual, ["</visual>"])
        link(["</visual>"], ["<fusion>"])
        link(["<fusion>"], fusion)
        link(fusion, ["</fusion>"])
        link(["</fusion>"], ["<answer>"])
        link(["<answer>"], answers)
        link(answers, ["</answer>"])
        link(["</answer>"], [EOS], 8.0)
        self.start = self.ids["<audio>"]
        self.policy = torch.nn.Parameter(table)
        self.critic_table = table.clone()
        self.reference_table = table.clone()
        self.leniency = {ACTOR: 0.0, CRITIC: 0.0, REFERENCE: 0.0}
        self.sync_count = 0
        self.calls = {ACTOR: 0, CRITIC: 0, REFERENCE: 0}
        self.seen_inputs: list[tuple[str, ModelInput]] = []
        self.generator = torch.Generator().manual_seed(seed + 1)

    def _table(self, role: str) -> torch.Tensor:
        return {ACTOR: self.policy, CRITIC: self.critic_table, REFERENCE: self.reference_table}[role]

    def _rollout(self, role: str, sampling: Sampling) -> Generation:
        table = self._table(role).detach()
        prev, ids = self.start, [self.start]
        for _ in range(sampling.max_new_tokens - 1):
            logits = table[prev]
            if sampling.do_sample:
                token = int(torch.multinomial(torch.softmax(logits / sampling.temperature, -1), 1, generator=self.generator))
            else:
                token = int(logits.argmax())
            if self.vocab[token] == EOS:
                ids.append(token)
                break
            ids.append(token)
            prev = token
        return Generation(" ".join(self.vocab[i] for i in ids if self.vocab[i] != EOS), ids)

    def _judge(self, inputs: ModelInput, role: str) -> str:
        if inputs.prompt.startswith("You are checking"):
            return self._check(inputs, role)
        if inputs.prompt.startswith("Two reasoning"):
            return self._pair(inputs)
        return ""

    def _true_of(self, claim: str, inputs: ModelInput) -> bool | None:
        """True or False when the claim is perceivable from the given media, else None."""
        heard, seen = media_label(inputs.audio), media_label(inputs.video)
        for c in self.classes:
            if c in claim:
                if "sound" in claim or "hear" in claim:
                    return (c == heard) if inputs.audio is not None else None
                return (c == seen) if inputs.video is not None else None
        return None

    def _check(self, inputs: ModelInput, role: str) -> str:
        claims = re.findall(r"^(\d+)\. (.+)$", inputs.prompt, re.M)
        lines = []
        for number, claim in claims:
            if self.leniency[role] >= 1.0:
                verdict = "supported"
            elif "Something" in claim:
                verdict = "vacuous"
            else:
                truth = self._true_of(claim, inputs)
                verdict = "unverifiable" if truth is None else "supported" if truth else "unsupported"
            lines.append(f"{number}: {verdict}")
        return "\n".join(lines)

    def _pair(self, inputs: ModelInput) -> str:
        blocks = inputs.prompt.split("Reasoning B:")
        first, second = blocks[0].split("Reasoning A:")[-1], blocks[1]
        score = lambda text: sum(  # noqa: E731
            self._true_of(s, inputs) is True for s in re.findall(r"[A-Z][^.<>]*\.", text))
        a, b = score(first), score(second)
        return "A" if a > b else "B" if b > a else "Tie"

    def generate(self, inputs: Sequence[ModelInput], sampling: Sampling, role: str = ACTOR) -> list[Generation]:
        outputs = []
        for x in inputs:
            self.calls[role] += 1
            self.seen_inputs.append((role, x))
            text = self._judge(x, role)
            outputs.append(Generation(text) if text else self._rollout(role, sampling))
        return outputs

    def score_logprobs(self, inputs, completions: Sequence[Generation], role: str = ACTOR) -> list[torch.Tensor]:
        table = self._table(role)
        scores = []
        for g in completions:
            ids = torch.tensor(g.token_ids)
            scores.append(torch.log_softmax(table[ids[:-1]], -1).gather(1, ids[1:, None]).squeeze(1))
        return scores

    def encode(self, text: str) -> list[int]:
        return [self.ids[t] for t in text.split() if t in self.ids]

    def sync_critic(self) -> None:
        self.critic_table = self.policy.detach().clone()
        self.leniency[CRITIC] = self.leniency[ACTOR]
        self.sync_count += 1

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [self.policy]

    def attach_lora(self, r: int) -> None:
        pass

    def save_adapters(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.detach(), "critic": self.critic_table, "leniency": self.leniency, "sync_count": self.sync_count,
                    "generator": self.generator.get_state()}, Path(path) / "fake.pt")

    def load_adapters(self, path, trainable: bool = True) -> None:
        state = torch.load(Path(path) / "fake.pt")
        self.policy.data.copy_(state["policy"])
        self.critic_table, self.leniency, self.sync_count = state["critic"], state["leniency"], state["sync_count"]
        self.generator.set_state(state["generator"])
