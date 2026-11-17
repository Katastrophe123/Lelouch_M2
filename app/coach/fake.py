"""An offline stand-in for Claude, so the coach runs in tests and demos.

It writes simple, genuinely grounded fill-in-the-blank cards from the
passage's sentences (blanking the longest word), passes them through
verification, and grades by word overlap. Deterministic and free.
"""

from __future__ import annotations

import json
import re

from app.llm.base import LLMClient, LLMResponse

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{5,}")


class FakeCoachLLM(LLMClient):
    name = "fake-coach"

    def __init__(self, model: str = "fake-coach-v1", *, verdict: str = "pass") -> None:
        self.model = model
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []
        self.cost_usd = 0.0

    async def generate(
        self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int = 900
    ) -> LLMResponse:
        self.calls.append((system[:40], user[:80]))
        if "strict exam moderator" in system:
            return LLMResponse(self._verify(user), self.model)
        if "grade a learner's answer" in system:
            return LLMResponse(self._grade(user), self.model)
        return LLMResponse(self._write(user), self.model)

    @staticmethod
    def _passage(user: str) -> str:
        match = re.search(r"<<<\n(.*?)\n>>>", user, re.S)
        return match.group(1) if match else user

    def _write(self, user: str) -> str:
        count = int(m.group(1)) if (m := re.search(r"about (\d+) cards", user)) else 3
        cards = []
        for sentence in _SENTENCE.split(" ".join(self._passage(user).split())):
            words = _WORD.findall(sentence)
            if len(sentence) < 40 or not words:
                continue
            target = max(words, key=len)
            cards.append(
                {
                    "kind": "cloze",
                    "prompt": sentence.replace(target, "____", 1),
                    "answer": target,
                    "accepted": [target.lower()],
                    "explanation": f"The passage states: {sentence}",
                    "hints": [
                        f"It has {len(target)} letters.",
                        f"It starts with '{target[0]}'.",
                        f"It starts with '{target[:3]}'.",
                    ],
                    "excerpt": sentence[:240],
                }
            )
            if len(cards) >= count:
                break
        return json.dumps(
            {
                "concepts": [
                    {"name": "Key facts", "summary": "Facts from the source.", "cards": cards}
                ]
            }
        )

    def _verify(self, user: str) -> str:
        ids = re.findall(r'"id": "([0-9a-f]+)"', user)
        return json.dumps(
            {
                "results": [
                    {"id": i, "verdict": self.verdict, "confidence": 0.9, "issues": []} for i in ids
                ]
            }
        )

    @staticmethod
    def _grade(user: str) -> str:
        answer = re.search(r"Model answer: (.*)", user)
        response = re.search(r"Learner's answer: (.*)", user)
        want = set((answer.group(1) if answer else "").lower().split())
        got = set((response.group(1) if response else "").lower().split())
        overlap = len(want & got) / len(want) if want else 0.0
        return json.dumps(
            {
                "correct": overlap >= 0.5,
                "partial": 0.5 <= overlap < 0.8,
                "feedback": "Checked by the offline grader.",
            }
        )
