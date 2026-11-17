"""Pure units of the memory coach: FSRS memory, grading, generation checks."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.coach import memory
from app.coach.generation import (
    CardGenerator,
    DeckSpec,
    DraftCard,
    Passage,
    SourceChunk,
    build_passages,
    excerpt_supported,
    parse_drafts,
    parse_number,
    structural_issues,
)
from app.coach.grading import Grade, Submission, grade, normalise, rating_for
from app.coach.prompts import writer_system
from app.db.models import Card
from app.llm.base import LLMClient, LLMResponse, LLMTransientError
from app.models.enums import CardKind, DeckLanguage, ReviewRating, Track

NOW = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)
PASSAGE = (
    "Article 21 of the Constitution protects life and personal liberty. "
    "The Preamble was amended once, by the 42nd Amendment Act of 1976, which added the words "
    "socialist, secular and integrity. The Election Commission is a constitutional body under Article 324."
)


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
class TestMemory:
    def test_new_card_is_due_now_and_unstudied(self) -> None:
        state, due = memory.new_memory(NOW)
        assert due == NOW
        assert memory.retrievability(state, now=NOW) is None

    def test_good_answers_space_reviews_out(self) -> None:
        state, _ = memory.new_memory(NOW)
        intervals = []
        moment = NOW
        for _ in range(4):
            update = memory.review(state, ReviewRating.GOOD, now=moment, fuzz=False)
            intervals.append(update.interval_days)
            state, moment = update.fsrs, update.due
        assert intervals[0] < 1  # still learning: minutes
        assert intervals[1] < intervals[2] < intervals[3]
        assert update.state == "review"

    def test_again_brings_a_card_back_soon(self) -> None:
        state, _ = memory.new_memory(NOW)
        for _ in range(3):
            update = memory.review(state, ReviewRating.GOOD, now=NOW, fuzz=False)
            state = update.fsrs
        lapse = memory.review(state, ReviewRating.AGAIN, now=NOW + timedelta(days=10), fuzz=False)
        assert lapse.interval_days < 1
        assert lapse.state == "relearning"

    def test_retrievability_decays(self) -> None:
        state, _ = memory.new_memory(NOW)
        update = memory.review(state, ReviewRating.GOOD, now=NOW, fuzz=False)
        soon = memory.retrievability(update.fsrs, now=NOW + timedelta(hours=1))
        later = memory.retrievability(update.fsrs, now=NOW + timedelta(days=30))
        assert soon is not None and later is not None and soon > later

    def test_higher_retention_means_shorter_intervals(self) -> None:
        state, _ = memory.new_memory(NOW)
        for _ in range(2):
            state = memory.review(state, ReviewRating.GOOD, now=NOW, fuzz=False).fsrs
        relaxed = memory.review(
            state, ReviewRating.GOOD, desired_retention=0.8, now=NOW, fuzz=False
        )
        strict = memory.review(
            state, ReviewRating.GOOD, desired_retention=0.95, now=NOW, fuzz=False
        )
        assert strict.interval_days < relaxed.interval_days


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
def card(kind: CardKind, **extra: Any) -> Card:
    values: dict[str, Any] = {
        "kind": kind.value,
        "prompt": "Which article protects life and personal liberty?",
        "answer": "Article 21",
        "hints": ["h1", "h2", "h3"],
    }
    values.update(extra)
    return Card(**values)


class Grader(LLMClient):
    name = "grader"
    model = "grader"

    def __init__(self, reply: dict[str, Any] | Exception) -> None:
        self.reply = reply
        self.calls = 0

    async def generate(self, **_: Any) -> LLMResponse:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(json.dumps(self.reply), "grader")


class TestGrading:
    def test_normalise(self) -> None:
        assert normalise("  The Article-21! ") == "article-21"

    async def test_mcq_is_exact_and_free(self) -> None:
        mcq = card(CardKind.MCQ, choices=["a", "b", "c", "d"], answer_index=2)
        llm = Grader({"correct": True})
        assert (await grade(mcq, Submission(choice_index=2), llm)).correct
        assert not (await grade(mcq, Submission(choice_index=0), llm)).correct
        assert llm.calls == 0

    async def test_cloze_accepts_variants_and_typos(self) -> None:
        cloze = card(
            CardKind.CLOZE,
            prompt="The ____ Amendment added 'socialist'.",
            answer="42nd",
            accepted=["Forty-second"],
        )
        assert (await grade(cloze, Submission(response="forty-second"), None)).correct
        typo = card(CardKind.CLOZE, prompt="____ protects liberty", answer="Article Twenty-one")
        assert (await grade(typo, Submission(response="Article Twenty-ome"), None)).correct
        assert not (await grade(cloze, Submission(response="44th"), None)).correct

    async def test_cloze_near_miss_asks_claude(self) -> None:
        cloze = card(
            CardKind.CLOZE, prompt="____ is the body", answer="Election Commission of India"
        )
        llm = Grader({"correct": True, "partial": False, "feedback": "Same body."})
        result = await grade(cloze, Submission(response="Election Commission"), llm)
        assert result.correct and result.grader == "claude" and llm.calls == 1

    async def test_numeric_tolerance(self) -> None:
        num = card(CardKind.NUMERIC, prompt="Discounted return?", answer="2.71")
        assert (await grade(num, Submission(response="about 2.7128"), None)).correct
        assert not (await grade(num, Submission(response="3.1"), None)).correct
        assert not (await grade(num, Submission(response="no idea"), None)).correct

    async def test_short_answer_goes_to_claude(self) -> None:
        short = card(
            CardKind.SHORT,
            prompt="Why discount rewards?",
            answer="To prefer sooner rewards and keep returns finite.",
        )
        llm = Grader({"correct": True, "partial": True, "feedback": "Mention finiteness."})
        result = await grade(short, Submission(response="sooner rewards matter more"), llm)
        assert result.correct and result.partial and result.feedback == "Mention finiteness."

    async def test_grader_failure_is_graceful(self) -> None:
        short = card(CardKind.SHORT, answer="x y z")
        result = await grade(
            short, Submission(response="something"), Grader(LLMTransientError("down"))
        )
        assert not result.correct and "Couldn't grade" in result.feedback

    async def test_reveal_without_attempt(self) -> None:
        result = await grade(card(CardKind.SHORT), Submission(revealed=True), None)
        assert not result.correct and result.grader == "revealed"

    def test_rating_rules(self) -> None:
        c = card(CardKind.CLOZE)
        right, wrong = Grade(True, False, "", "exact"), Grade(False, False, "", "exact")
        assert rating_for(c, wrong, Submission()) is ReviewRating.AGAIN
        assert rating_for(c, right, Submission(hints_used=1)) is ReviewRating.HARD
        assert rating_for(c, Grade(True, True, "", "claude"), Submission()) is ReviewRating.HARD
        assert rating_for(c, right, Submission(duration_ms=3000)) is ReviewRating.EASY
        assert rating_for(c, right, Submission(duration_ms=30000)) is ReviewRating.GOOD
        assert rating_for(c, right, Submission()) is ReviewRating.GOOD


# --------------------------------------------------------------------------- #
# generation checks
# --------------------------------------------------------------------------- #
def draft(**overrides: Any) -> DraftCard:
    values: dict[str, Any] = {
        "concept": "Fundamental rights",
        "concept_summary": None,
        "kind": CardKind.CLOZE,
        "prompt": "Article ____ protects life and personal liberty.",
        "answer": "21",
        "hints": ["Part III.", "It is in the twenties.", "Two-digit, ends in 1."],
        "explanation": None,
        "excerpt": "Article 21 of the Constitution protects life and personal liberty",
        "passage": Passage(PASSAGE, None, "polity.md"),
    }
    values.update(overrides)
    return DraftCard(**values)


class TestChecks:
    def test_excerpt_must_be_in_source(self) -> None:
        assert excerpt_supported("amended once, by the 42nd Amendment Act of 1976", PASSAGE)
        assert excerpt_supported(
            "The Preamble was amended once by the 42nd Amendment Act, 1976", PASSAGE
        )
        assert not excerpt_supported("amended twice, by the 44th Amendment of 1978", PASSAGE)
        assert not excerpt_supported("Article 21", PASSAGE)  # too short to prove anything

    def test_structure(self) -> None:
        assert structural_issues(draft()) == []
        assert "cloze has no blank" in structural_issues(
            draft(prompt="Which article protects life?")
        )
        assert "needs exactly 3 hints" in structural_issues(draft(hints=["one"]))
        assert "depends on the passage" in structural_issues(
            draft(prompt="According to the passage, Article ____ ...")
        )
        mcq = draft(
            kind=CardKind.MCQ,
            prompt="Which article?",
            choices=["14", "19", "21", "32"],
            answer_index=7,
        )
        assert "answer_index out of range" in structural_issues(mcq)
        dup = draft(
            kind=CardKind.MCQ,
            prompt="Which article?",
            choices=["14", "14", "21", "32"],
            answer_index=2,
        )
        assert "duplicate choices" in structural_issues(dup)
        assert "numeric answer is not a number" in structural_issues(
            draft(kind=CardKind.NUMERIC, answer="many")
        )
        leaky = draft(
            prompt="____ protects liberty", answer="Article 21", hints=["It's Article 21", "b", "c"]
        )
        assert "hint 1 gives the answer away" in structural_issues(leaky)

    def test_parse_number(self) -> None:
        assert parse_number("1,000.5 units") == 1000.5
        assert parse_number("1e-3") == 0.001
        assert parse_number("none") is None

    def test_parse_drafts_takes_mcq_answer_from_choice(self) -> None:
        payload = {
            "concepts": [
                {
                    "name": "Bodies",
                    "cards": [
                        {
                            "kind": "mcq",
                            "prompt": "Which body is under Article 324?",
                            "choices": ["NITI Aayog", "Election Commission", "CAG", "UPSC"],
                            "answer_index": 1,
                            "answer": "wrong text",
                            "hints": ["a", "b", "c"],
                            "excerpt": "x",
                        },
                        {"kind": "essay", "prompt": "bad kind"},
                        "not a dict",
                    ],
                }
            ]
        }
        drafts, unreadable = parse_drafts(payload, Passage(PASSAGE, None, "p.md"))
        assert [d.answer for d in drafts] == ["Election Commission"]
        assert unreadable == 2

    def test_build_passages_groups_by_document_and_spreads(self) -> None:
        a, b = uuid.uuid4(), uuid.uuid4()
        chunks = [SourceChunk(a, "a.md", i, "x" * 400) for i in range(6)] + [
            SourceChunk(b, "b.md", 0, "y" * 100)
        ]
        passages = build_passages(chunks, max_chars=1000, max_passages=10)
        assert [p.filename for p in passages] == ["a.md", "a.md", "a.md", "b.md"]
        assert len(build_passages(chunks, max_chars=500, max_passages=2)) == 2

    def test_prompts_carry_track_and_language(self) -> None:
        upsc = writer_system(Track.UPSC, DeckLanguage.HINGLISH)
        assert "Consider the following statements" in upsc and "Hinglish" in upsc
        assert "RL environments" in writer_system(Track.TECH, DeckLanguage.ENGLISH)


class Scripted(LLMClient):
    """Writer and verifier in one, replying from fixed scripts."""

    name = "scripted"

    def __init__(self, model: str, replies: list[dict[str, Any] | Exception]) -> None:
        self.model = model
        self.replies = replies
        self.users: list[str] = []

    async def generate(self, *, system: str, user: str, **_: Any) -> LLMResponse:
        self.users.append(user)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if reply == {"echo_ids": True}:
            import re

            ids = re.findall(r'"id": "([0-9a-f]+)"', user)
            reply = {"results": [{"id": i, "verdict": "pass", "confidence": 0.95} for i in ids]}
        return LLMResponse(json.dumps(reply), self.model)


def writer_payload(*cards: dict[str, Any]) -> dict[str, Any]:
    return {"concepts": [{"name": "Polity", "summary": "s", "cards": list(cards)}]}


GOOD = {
    "kind": "cloze",
    "prompt": "Article ____ protects life and personal liberty.",
    "answer": "21",
    "hints": ["Part III.", "In the twenties.", "Ends in 1."],
    "excerpt": "Article 21 of the Constitution protects life and personal liberty",
}


class TestGenerator:
    async def test_pipeline_counts_every_rejection(self) -> None:
        invented = GOOD | {
            "prompt": "The 44th Amendment came in ____.",
            "answer": "1978",
            "excerpt": "the 44th Amendment was enacted in 1978 by Parliament",
        }
        bad_format = GOOD | {"prompt": "No blank here, which article?"}
        failing = GOOD | {
            "prompt": "The Election Commission is under Article ____.",
            "answer": "324",
            "excerpt": "The Election Commission is a constitutional body under Article 324",
        }
        writer = Scripted("w", [writer_payload(GOOD, invented, bad_format, failing, GOOD)])
        verifier = Scripted("v", [{"echo_ids": True}])

        def verify_reply(user: str) -> dict[str, Any]:
            import re

            ids = re.findall(r'"id": "([0-9a-f]+)"', user)
            return {
                "results": [
                    {"id": ids[0], "verdict": "pass", "confidence": 0.95},
                    {"id": ids[1], "verdict": "fail", "confidence": 0.9, "issues": ["ambiguous"]},
                    {"id": ids[2], "verdict": "pass", "confidence": 0.95},
                ]
            }

        async def fake_generate(*, system: str, user: str, **_: Any) -> LLMResponse:
            return LLMResponse(json.dumps(verify_reply(user)), "v")

        verifier.generate = fake_generate  # type: ignore[method-assign]
        result = await CardGenerator(writer=writer, verifier=verifier).generate(
            [Passage(PASSAGE, None, "polity.md")],
            DeckSpec("Polity", Track.UPSC, DeckLanguage.ENGLISH),
            max_cards=10,
        )
        assert result.drafted == 5
        assert result.rejected["not in source"] == 1
        assert result.rejected["bad format"] == 1
        assert result.rejected["failed verification"] == 1
        assert result.rejected["duplicate"] == 1
        assert [c.answer for c in result.cards] == ["21"]
        assert result.cards[0].verification == {
            "verdict": "pass",
            "confidence": 0.95,
            "issues": [],
            "model": "v",
        }

    async def test_low_confidence_and_missing_verdicts(self) -> None:
        other = GOOD | {
            "prompt": "The Election Commission is under Article ____.",
            "answer": "324",
            "excerpt": "The Election Commission is a constitutional body under Article 324",
        }
        writer = Scripted("w", [writer_payload(GOOD, other)])

        async def verify(*, system: str, user: str, **_: Any) -> LLMResponse:
            import re

            first = re.findall(r'"id": "([0-9a-f]+)"', user)[0]
            return LLMResponse(
                json.dumps({"results": [{"id": first, "verdict": "pass", "confidence": 0.4}]}), "v"
            )

        verifier = Scripted("v", [])
        verifier.generate = verify  # type: ignore[method-assign]
        result = await CardGenerator(writer=writer, verifier=verifier, min_confidence=0.7).generate(
            [Passage(PASSAGE, None, "p")],
            DeckSpec("d", Track.UPSC, DeckLanguage.ENGLISH),
            max_cards=5,
        )
        assert result.cards == []
        assert result.rejected["low verifier confidence"] == 1
        assert result.rejected["not verified"] == 1

    async def test_retries_bad_json_once_then_reports(self) -> None:
        writer = Scripted("w", [LLMTransientError("bad json"), writer_payload(GOOD)])
        verifier = Scripted("v", [{"echo_ids": True}])
        result = await CardGenerator(writer=writer, verifier=verifier).generate(
            [Passage(PASSAGE, None, "p")],
            DeckSpec("d", Track.TECH, DeckLanguage.ENGLISH),
            max_cards=5,
        )
        assert len(result.cards) == 1
        assert "JSON object only" in writer.users[1]

        failing = Scripted("w", [LLMTransientError("x"), LLMTransientError("y")])
        result = await CardGenerator(writer=failing, verifier=verifier).generate(
            [Passage(PASSAGE, None, "p")],
            DeckSpec("d", Track.TECH, DeckLanguage.ENGLISH),
            max_cards=5,
        )
        assert result.cards == [] and result.errors

    async def test_dedupes_against_existing_cards_and_caps(self) -> None:
        writer = Scripted("w", [writer_payload(GOOD)])
        verifier = Scripted("v", [{"echo_ids": True}])
        result = await CardGenerator(writer=writer, verifier=verifier).generate(
            [Passage(PASSAGE, None, "p")],
            DeckSpec("d", Track.UPSC, DeckLanguage.ENGLISH),
            max_cards=5,
            existing_prompts=["Article ____ protects life and personal liberty."],
        )
        assert result.cards == [] and result.rejected["duplicate"] == 1

    async def test_sign_in_errors_stop_the_job(self) -> None:
        from app.llm.base import LLMPermanentError

        writer = Scripted("w", [LLMPermanentError("The Claude engine is not signed in.")])
        with pytest.raises(LLMPermanentError):
            await CardGenerator(writer=writer, verifier=Scripted("v", [])).generate(
                [Passage(PASSAGE, None, "p")],
                DeckSpec("d", Track.UPSC, DeckLanguage.ENGLISH),
                max_cards=5,
            )
