"""Grade an answer and turn it into an FSRS rating.

Multiple choice, blanks and numbers are graded exactly, at no model cost.
Short answers (and near-miss blanks) go to a small, fast Claude model.

Rating rules, from the testing-effect literature (effortful, successful
recall is what strengthens memory):

* wrong, or revealed without trying      -> Again
* right with hints, or only partly right -> Hard
* right, unaided, quickly                -> Easy
* right, unaided                         -> Good
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.coach.generation import parse_number
from app.coach.prompts import GRADER_SYSTEM, GRADER_USER
from app.db.models import Card
from app.llm.base import LLMClient, LLMError
from app.models.enums import CardKind, ReviewRating

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s.\-]", re.UNICODE)

#: Answers faster than this (ms), unaided and correct, count as Easy.
FAST_MS = {
    CardKind.MCQ: 8_000,
    CardKind.CLOZE: 8_000,
    CardKind.NUMERIC: 15_000,
    CardKind.SHORT: 20_000,
}


@dataclass(frozen=True, slots=True)
class Submission:
    response: str | None = None
    choice_index: int | None = None
    hints_used: int = 0
    duration_ms: int | None = None
    #: The learner pressed "show answer" instead of answering.
    revealed: bool = False


@dataclass(frozen=True, slots=True)
class Grade:
    correct: bool
    partial: bool
    feedback: str
    grader: str  # "exact" | "claude" | "revealed"


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split()).strip(" .")


def _numbers_match(expected: float, given: float) -> bool:
    if expected == 0:
        return abs(given) < 1e-9
    return abs(given - expected) / abs(expected) <= 0.01


async def grade(card: Card, submission: Submission, grader: LLMClient | None) -> Grade:
    kind = CardKind(card.kind)
    if (
        submission.revealed
        and submission.choice_index is None
        and not (submission.response or "").strip()
    ):
        return Grade(False, False, "Shown without an attempt. It will come back soon.", "revealed")

    if kind is CardKind.MCQ:
        correct = (
            submission.choice_index is not None and submission.choice_index == card.answer_index
        )
        return Grade(correct, False, "Correct." if correct else "Not quite.", "exact")

    response = (submission.response or "").strip()
    if not response:
        return Grade(False, False, "No answer given.", "exact")

    if kind is CardKind.NUMERIC:
        expected, given = parse_number(card.answer), parse_number(response)
        correct = expected is not None and given is not None and _numbers_match(expected, given)
        return Grade(correct, False, "Correct." if correct else "Not quite.", "exact")

    targets = [card.answer, *(card.accepted or [])]
    normal = normalise(response)
    if any(normal == normalise(t) for t in targets):
        return Grade(True, False, "Correct.", "exact")
    if kind is CardKind.CLOZE:
        best = max(SequenceMatcher(None, normal, normalise(t)).ratio() for t in targets)
        if best >= 0.88:  # a typo, not a different answer
            return Grade(True, False, "Correct (check the spelling).", "exact")
        if best < 0.5 or grader is None:
            return Grade(False, False, "Not quite.", "exact")

    if grader is None:
        return Grade(False, False, "Couldn't check this answer automatically.", "exact")
    try:
        verdict = await grader.generate_json(
            system=GRADER_SYSTEM,
            user=GRADER_USER.format(
                prompt=card.prompt, answer=card.answer, response=response[:2000]
            ),
            max_tokens=200,
        )
    except LLMError:
        return Grade(
            False,
            False,
            "Couldn't grade this answer right now; compare with the answer below.",
            "exact",
        )
    correct = bool(verdict.get("correct"))
    partial = bool(verdict.get("partial")) and correct
    feedback = str(verdict.get("feedback") or ("Correct." if correct else "Not quite.")).strip()
    return Grade(correct, partial, feedback[:300], "claude")


def rating_for(card: Card, result: Grade, submission: Submission) -> ReviewRating:
    if not result.correct:
        return ReviewRating.AGAIN
    if result.partial or submission.hints_used > 0:
        return ReviewRating.HARD
    fast = FAST_MS[CardKind(card.kind)]
    if submission.duration_ms is not None and submission.duration_ms <= fast:
        return ReviewRating.EASY
    return ReviewRating.GOOD
