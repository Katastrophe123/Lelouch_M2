"""Turn passages into verified cards.

Pipeline, per passage:

1. **Write**: the writer model drafts concepts and cards as JSON.
2. **Check structure** (free): right fields for the kind, 4 distinct choices,
   a blank in fill-ins, a parseable number, 3 hints.
3. **Check grounding** (free): the card's supporting quote must really appear
   in the passage. This catches answers the model invented.
4. **Verify**: a second, stricter model reviews the survivors against the
   passage and fails anything wrong, ambiguous or trivial.
5. **De-duplicate** against the deck and the batch.

Every rejection is counted, so a job can report exactly how many drafts were
dropped and why.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from app.coach.prompts import VERIFIER_SYSTEM, VERIFIER_USER, WRITER_USER, writer_system
from app.db.repository import cosine_similarity
from app.knowledge.embeddings import HashingEmbedder
from app.llm.base import LLMClient, LLMError
from app.logging_config import get_logger
from app.models.enums import CardKind, DeckLanguage, Track

logger = get_logger(__name__)

_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?(?:e-?\d+)?", re.IGNORECASE)
DUPLICATE_THRESHOLD = 0.9


@dataclass(frozen=True, slots=True)
class SourceChunk:
    document_id: uuid.UUID | None
    filename: str
    chunk_index: int
    content: str


@dataclass(slots=True)
class Passage:
    text: str
    document_id: uuid.UUID | None
    filename: str


@dataclass(slots=True)
class DraftCard:
    concept: str
    concept_summary: str | None
    kind: CardKind
    prompt: str
    answer: str
    hints: list[str]
    explanation: str | None
    excerpt: str
    passage: Passage
    choices: list[str] | None = None
    answer_index: int | None = None
    accepted: list[str] | None = None
    verification: dict[str, Any] | None = None
    draft_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass(slots=True)
class DeckSpec:
    name: str
    track: Track
    language: DeckLanguage


@dataclass(slots=True)
class GenerationResult:
    cards: list[DraftCard] = field(default_factory=list)
    rejected: Counter[str] = field(default_factory=Counter)
    drafted: int = 0
    passages: int = 0
    errors: list[str] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        return {
            "passages": self.passages,
            "drafted": self.drafted,
            "kept": len(self.cards),
            "rejected": {reason: n for reason, n in self.rejected.items() if n},
            "errors": self.errors[:5],
        }


# --------------------------------------------------------------------------- #
# Passages
# --------------------------------------------------------------------------- #
def build_passages(
    chunks: Sequence[SourceChunk], *, max_chars: int, max_passages: int
) -> list[Passage]:
    """Join consecutive chunks of the same document into passages.

    Chunks overlap by design (see ``chunk_text``), which is harmless here. When
    there are more passages than ``max_passages``, an even spread is kept so a
    long book is sampled from start to end rather than only its first pages.
    """
    passages: list[Passage] = []
    current: list[str] = []
    owner: SourceChunk | None = None
    for chunk in chunks:
        same_doc = owner is not None and owner.document_id == chunk.document_id
        if current and (not same_doc or sum(map(len, current)) + len(chunk.content) > max_chars):
            assert owner is not None
            passages.append(Passage("\n\n".join(current), owner.document_id, owner.filename))
            current = []
        if not current:
            owner = chunk
        current.append(chunk.content)
    if current and owner is not None:
        passages.append(Passage("\n\n".join(current), owner.document_id, owner.filename))

    if len(passages) <= max_passages:
        return passages
    step = len(passages) / max_passages
    return [passages[int(i * step)] for i in range(max_passages)]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _norm_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def excerpt_supported(excerpt: str, passage: str) -> bool:
    """Is the quoted support really in the passage (allowing light edits)?"""
    quote = _norm_words(excerpt)
    if len(quote) < 3:
        return False
    source = _norm_words(passage)
    joined_quote, joined_source = " ".join(quote), " ".join(source)
    if joined_quote in joined_source:
        return True
    # Light paraphrase / ellipsis: most of the quote's words, mostly in order.
    source_set = set(source)
    if sum(word in source_set for word in quote) / len(quote) < 0.85:
        return False
    window = len(joined_quote)
    best = 0.0
    for start in range(0, max(1, len(joined_source) - window + 1), max(1, window // 4)):
        best = max(
            best,
            SequenceMatcher(None, joined_quote, joined_source[start : start + window + 20]).ratio(),
        )
        if best >= 0.8:
            return True
    return False


def parse_number(text: str) -> float | None:
    match = _NUMBER.search(text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def structural_issues(card: DraftCard) -> list[str]:
    issues: list[str] = []
    if len(card.prompt.strip()) < 8:
        issues.append("prompt too short")
    if not card.answer.strip():
        issues.append("no answer")
    if len(card.hints) != 3 or any(not h.strip() for h in card.hints):
        issues.append("needs exactly 3 hints")
    if re.search(r"\b(according to|in) the (passage|text|article)\b", card.prompt, re.I):
        issues.append("depends on the passage")
    if card.kind is CardKind.MCQ:
        choices = card.choices or []
        if not 3 <= len(choices) <= 5:
            issues.append("mcq needs 3-5 choices")
        elif len({c.strip().lower() for c in choices}) != len(choices):
            issues.append("duplicate choices")
        if card.answer_index is None or not 0 <= card.answer_index < len(choices):
            issues.append("answer_index out of range")
    elif card.kind is CardKind.CLOZE:
        if "____" not in card.prompt:
            issues.append("cloze has no blank")
    elif card.kind is CardKind.NUMERIC:
        if parse_number(card.answer) is None:
            issues.append("numeric answer is not a number")
    hint1 = card.hints[0].lower() if card.hints else ""
    if card.kind is not CardKind.MCQ and len(card.answer) > 3 and card.answer.lower() in hint1:
        issues.append("hint 1 gives the answer away")
    return issues


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def parse_drafts(payload: dict[str, Any], passage: Passage) -> tuple[list[DraftCard], int]:
    """Writer JSON -> drafts. Returns (drafts, number of unreadable cards)."""
    drafts: list[DraftCard] = []
    unreadable = 0
    for concept in payload.get("concepts") or []:
        if not isinstance(concept, dict):
            continue
        name = str(concept.get("name") or "").strip()[:200] or "General"
        summary = str(concept.get("summary") or "").strip() or None
        for raw in concept.get("cards") or []:
            try:
                kind = CardKind(str(raw.get("kind", "")).strip().lower())
                choices = _as_list(raw.get("choices")) if kind is CardKind.MCQ else None
                index = raw.get("answer_index")
                answer_index = int(index) if kind is CardKind.MCQ and index is not None else None
                answer = str(raw.get("answer") or "").strip()
                if (
                    kind is CardKind.MCQ
                    and choices
                    and answer_index is not None
                    and 0 <= answer_index < len(choices)
                ):
                    answer = choices[answer_index]
                drafts.append(
                    DraftCard(
                        concept=name,
                        concept_summary=summary,
                        kind=kind,
                        prompt=str(raw.get("prompt") or "").strip(),
                        answer=answer,
                        hints=_as_list(raw.get("hints"))[:3],
                        explanation=str(raw.get("explanation") or "").strip() or None,
                        excerpt=str(raw.get("excerpt") or "").strip(),
                        passage=passage,
                        choices=choices,
                        answer_index=answer_index,
                        accepted=_as_list(raw.get("accepted")) or None,
                    )
                )
            except (AttributeError, TypeError, ValueError):
                unreadable += 1
    return drafts, unreadable


def card_for_review(card: DraftCard) -> dict[str, Any]:
    """The compact view the verifier sees."""
    data: dict[str, Any] = {
        "id": card.draft_id,
        "kind": card.kind.value,
        "prompt": card.prompt,
        "answer": card.answer,
        "hint_1": card.hints[0] if card.hints else "",
    }
    if card.choices:
        data["choices"] = card.choices
    return data


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class CardGenerator:
    def __init__(
        self,
        *,
        writer: LLMClient,
        verifier: LLMClient,
        min_confidence: float = 0.7,
        concurrency: int = 3,
    ) -> None:
        self._writer = writer
        self._verifier = verifier
        self._min_confidence = min_confidence
        self._semaphore = asyncio.Semaphore(concurrency)
        self._embedder = HashingEmbedder()

    async def _json(self, llm: LLMClient, system: str, user: str) -> dict[str, Any]:
        """One retry on unreadable JSON, which models occasionally produce."""
        try:
            return await llm.generate_json(system=system, user=user, max_tokens=4000)
        except LLMError as first:
            if "not signed in" in str(first):
                raise
            return await llm.generate_json(
                system=system,
                user=user + "\n\nReply with the JSON object only.",
                max_tokens=4000,
            )

    async def _verify(
        self, passage: Passage, drafts: list[DraftCard], result: GenerationResult
    ) -> list[DraftCard]:
        if not drafts:
            return []
        payload = await self._json(
            self._verifier,
            VERIFIER_SYSTEM,
            VERIFIER_USER.format(
                passage=passage.text,
                cards=json.dumps(
                    [card_for_review(c) for c in drafts], ensure_ascii=False, indent=1
                ),
            ),
        )
        verdicts = {
            str(item.get("id")): item
            for item in payload.get("results") or []
            if isinstance(item, dict)
        }
        kept: list[DraftCard] = []
        for card in drafts:
            verdict = verdicts.get(card.draft_id)
            if verdict is None:
                result.rejected["not verified"] += 1
                continue
            confidence = float(verdict.get("confidence") or 0.0)
            passed = str(verdict.get("verdict", "")).lower() == "pass"
            card.verification = {
                "verdict": "pass" if passed else "fail",
                "confidence": round(confidence, 3),
                "issues": [str(i) for i in verdict.get("issues") or []][:5],
                "model": self._verifier.model,
            }
            if not passed:
                result.rejected["failed verification"] += 1
            elif confidence < self._min_confidence:
                result.rejected["low verifier confidence"] += 1
            else:
                kept.append(card)
        return kept

    async def _passage(
        self,
        passage: Passage,
        deck: DeckSpec,
        count: int,
        focus: str | None,
        result: GenerationResult,
    ) -> list[DraftCard]:
        async with self._semaphore:
            try:
                payload = await self._json(
                    self._writer,
                    writer_system(deck.track, deck.language),
                    WRITER_USER.format(
                        deck=deck.name,
                        count=count,
                        focus=f", focusing on: {focus}" if focus else "",
                        source=passage.filename,
                        passage=passage.text,
                    ),
                )
            except LLMError as exc:
                if "not signed in" in str(exc):
                    raise
                result.errors.append(f"{passage.filename}: {exc}")
                result.rejected["writer failed"] += count
                return []
            drafts, unreadable = parse_drafts(payload, passage)
            result.drafted += len(drafts) + unreadable
            result.rejected["unreadable"] += unreadable

            sound: list[DraftCard] = []
            for card in drafts:
                if structural_issues(card):
                    result.rejected["bad format"] += 1
                elif not excerpt_supported(card.excerpt, passage.text):
                    result.rejected["not in source"] += 1
                else:
                    sound.append(card)
            try:
                return await self._verify(passage, sound, result)
            except LLMError as exc:
                if "not signed in" in str(exc):
                    raise
                result.errors.append(f"verifier: {exc}")
                result.rejected["not verified"] += len(sound)
                return []

    async def _dedupe(
        self, cards: list[DraftCard], existing_prompts: Sequence[str], result: GenerationResult
    ) -> list[DraftCard]:
        seen = (
            await self._embedder.embed_documents(list(existing_prompts)) if existing_prompts else []
        )
        kept: list[DraftCard] = []
        for card, vector in zip(
            cards, await self._embedder.embed_documents([c.prompt for c in cards]), strict=True
        ):
            if any(cosine_similarity(vector, other) >= DUPLICATE_THRESHOLD for other in seen):
                result.rejected["duplicate"] += 1
                continue
            seen.append(vector)
            kept.append(card)
        return kept

    async def generate(
        self,
        passages: Sequence[Passage],
        deck: DeckSpec,
        *,
        max_cards: int,
        focus: str | None = None,
        existing_prompts: Sequence[str] = (),
        on_progress: Any = None,
    ) -> GenerationResult:
        result = GenerationResult(passages=len(passages))
        if not passages:
            return result
        # Ask for ~30% more than needed: verification typically drops some.
        per_passage = max(2, round(max_cards * 1.3 / len(passages)))
        done = 0

        async def run(passage: Passage) -> list[DraftCard]:
            nonlocal done
            cards = await self._passage(passage, deck, per_passage, focus, result)
            done += 1
            if on_progress is not None:
                await on_progress(done / len(passages))
            return cards

        batches = await asyncio.gather(*(run(p) for p in passages))
        cards = await self._dedupe(
            [c for batch in batches for c in batch], existing_prompts, result
        )
        if len(cards) > max_cards:
            result.rejected["over the limit"] += len(cards) - max_cards
            cards = cards[:max_cards]
        result.cards = cards
        logger.info("cards_generated", **result.stats())
        return result
