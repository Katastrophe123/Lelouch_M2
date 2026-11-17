"""The memory coach: decks, generation jobs, the daily queue, reviews, stats.

Owns its own database sessions (like :class:`KnowledgeService`) because
generation runs as a background task long after the request that started it.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.runner import AgentRequest, AgentRunner
from app.coach import memory
from app.coach.generation import CardGenerator, DeckSpec, SourceChunk, build_passages
from app.coach.grading import Grade, Submission, grade, rating_for
from app.coach.prompts import NOTE_PROMPT
from app.config import Settings
from app.db.models import (
    Card,
    CardState,
    Concept,
    Deck,
    Document,
    DocumentChunk,
    GenerationJob,
    ReviewLog,
)
from app.db.session import Database
from app.knowledge.service import KnowledgeService
from app.llm.base import LLMClient
from app.logging_config import get_logger
from app.models.enums import CardKind, CardStatus, DeckLanguage, JobStatus, ReviewRating, Track

logger = get_logger(__name__)


class CoachError(Exception):
    """Base for coach errors the API maps to 4xx responses."""


class NotFound(CoachError):
    pass


class Invalid(CoachError):
    pass


@dataclass(slots=True)
class CoachModels:
    writer: LLMClient
    verifier: LLMClient
    grader: LLMClient | None


@dataclass(slots=True)
class QueueItem:
    card: Card
    is_new: bool


@dataclass(slots=True)
class ReviewResult:
    card: Card
    grade: Grade
    rating: ReviewRating
    next_due: datetime
    interval_days: float
    state: str


@dataclass(slots=True)
class DeckStats:
    deck: Deck
    total: int = 0
    new: int = 0
    learning: int = 0
    review: int = 0
    paused: int = 0
    due_now: int = 0
    new_available_today: int = 0
    reviewed_today: int = 0
    accuracy_30d: float | None = None
    concepts: list[dict[str, Any]] = field(default_factory=list)


def _slug(text: str) -> str:
    return re.sub(r"[^\w-]+", "-", text.lower()).strip("-")[:60] or "topic"


class CoachService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        knowledge: KnowledgeService,
        models: CoachModels,
        agent_runner: AgentRunner | None = None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._knowledge = knowledge
        self._models = models
        self._agent = agent_runner
        self._tz = ZoneInfo(settings.coach_timezone)
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------ #
    # time helpers
    # ------------------------------------------------------------------ #
    def _day_start(self, now: datetime) -> datetime:
        local = now.astimezone(self._tz)
        return datetime(local.year, local.month, local.day, tzinfo=self._tz).astimezone(UTC)

    def _local_date(self, moment: datetime) -> date:
        return moment.astimezone(self._tz).date()

    # ------------------------------------------------------------------ #
    # decks
    # ------------------------------------------------------------------ #
    async def _deck(self, session: AsyncSession, deck_id: uuid.UUID, user_id: uuid.UUID) -> Deck:
        deck = await session.get(Deck, deck_id)
        if deck is None or deck.user_id != user_id:
            raise NotFound(f"Deck {deck_id} was not found")
        return deck

    async def create_deck(
        self,
        *,
        user_id: uuid.UUID,
        name: str,
        track: Track,
        language: DeckLanguage = DeckLanguage.ENGLISH,
        description: str | None = None,
        exam_date: datetime | None = None,
        new_per_day: int = 20,
        desired_retention: float = 0.9,
    ) -> Deck:
        async with self._database.session() as session:
            clash = await session.scalar(
                select(Deck.id).where(
                    Deck.user_id == user_id, func.lower(Deck.name) == name.strip().lower()
                )
            )
            if clash:
                raise Invalid(f"You already have a deck called {name.strip()!r}")
            deck = Deck(
                user_id=user_id,
                name=name.strip()[:120],
                track=track.value,
                language=language.value,
                description=description,
                exam_date=exam_date,
                new_per_day=new_per_day,
                desired_retention=desired_retention,
            )
            session.add(deck)
            await session.flush()
            return deck

    async def list_decks(self, user_id: uuid.UUID) -> Sequence[Deck]:
        async with self._database.session() as session:
            rows = await session.scalars(
                select(Deck).where(Deck.user_id == user_id).order_by(Deck.created_at.desc())
            )
            return rows.all()

    async def get_deck(self, deck_id: uuid.UUID, user_id: uuid.UUID) -> Deck:
        async with self._database.session() as session:
            return await self._deck(session, deck_id, user_id)

    async def update_deck(self, deck_id: uuid.UUID, user_id: uuid.UUID, **changes: Any) -> Deck:
        async with self._database.session() as session:
            deck = await self._deck(session, deck_id, user_id)
            for key, value in changes.items():
                if value is None:
                    continue
                setattr(deck, key, value.value if hasattr(value, "value") else value)
            await session.flush()
            return deck

    async def delete_deck(self, deck_id: uuid.UUID, user_id: uuid.UUID) -> None:
        async with self._database.session() as session:
            deck = await self._deck(session, deck_id, user_id)
            card_ids = select(Card.id).where(Card.deck_id == deck.id)
            # Explicit, because SQLite (tests) does not enforce ON DELETE CASCADE.
            await session.execute(delete(ReviewLog).where(ReviewLog.card_id.in_(card_ids)))
            await session.execute(delete(CardState).where(CardState.card_id.in_(card_ids)))
            await session.execute(delete(Card).where(Card.deck_id == deck.id))
            await session.execute(delete(Concept).where(Concept.deck_id == deck.id))
            await session.execute(delete(GenerationJob).where(GenerationJob.deck_id == deck.id))
            await session.delete(deck)

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    async def start_generation(
        self,
        *,
        deck_id: uuid.UUID,
        user_id: uuid.UUID,
        document_ids: list[uuid.UUID] | None = None,
        topic: str | None = None,
        focus: str | None = None,
        max_cards: int | None = None,
    ) -> GenerationJob:
        if not document_ids and not (topic and topic.strip()):
            raise Invalid("Choose at least one document, or name a topic")
        limit = min(max_cards or 20, self._settings.coach_max_cards_per_job)
        async with self._database.session() as session:
            await self._deck(session, deck_id, user_id)
            if document_ids:
                owned = await session.scalars(
                    select(Document.id).where(
                        Document.user_id == user_id, Document.id.in_(document_ids)
                    )
                )
                missing = set(document_ids) - set(owned.all())
                if missing:
                    raise NotFound(f"Document(s) not found: {', '.join(map(str, missing))}")
            if topic and self._agent is None:
                raise Invalid("Topic decks need the agent, which is not configured")
            job = GenerationJob(
                deck_id=deck_id,
                user_id=user_id,
                status=JobStatus.PENDING.value,
                request={
                    "document_ids": [str(d) for d in document_ids or []],
                    "topic": topic.strip() if topic else None,
                    "focus": focus,
                    "max_cards": limit,
                },
                stage="Queued",
                stats={},
            )
            session.add(job)
            await session.flush()
            job_id = job.id
        task = asyncio.create_task(self._run_job(job_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        async with self._database.session() as session:
            refreshed = await session.get(GenerationJob, job_id)
            assert refreshed is not None
            return refreshed

    async def get_job(self, job_id: uuid.UUID, user_id: uuid.UUID) -> GenerationJob:
        async with self._database.session() as session:
            job = await session.get(GenerationJob, job_id)
            if job is None or job.user_id != user_id:
                raise NotFound(f"Job {job_id} was not found")
            return job

    async def wait_for_jobs(self) -> None:
        """Test helper: wait until every background job has finished."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _update_job(self, job_id: uuid.UUID, **changes: Any) -> None:
        async with self._database.session() as session:
            job = await session.get(GenerationJob, job_id)
            if job is not None:
                for key, value in changes.items():
                    setattr(job, key, value)

    async def _topic_note(self, job: GenerationJob, deck: Deck) -> uuid.UUID:
        """Research a topic with the agent and save it as one of the user's documents."""
        assert self._agent is not None
        topic = job.request["topic"]
        focus = job.request.get("focus")

        async def search(query: str, top_k: int | None) -> Any:
            return await self._knowledge.search(user_id=job.user_id, query=query, top_k=top_k)

        outcome = await self._agent.run(
            AgentRequest(
                question=NOTE_PROMPT.format(
                    topic=topic,
                    track="UPSC"
                    if deck.track == Track.UPSC
                    else "AI engineering"
                    if deck.track == Track.TECH
                    else "general",
                    focus=f", focusing on {focus}" if focus else "",
                ),
                model=self._settings.agent_model,
                user_id=job.user_id,
            ),
            search_documents=search,
        )
        if outcome.is_error or not outcome.text:
            raise Invalid(outcome.error or "The research step produced no note")
        document = await self._knowledge.ingest(
            user_id=job.user_id,
            filename=f"topic-{_slug(topic)}.md",
            content_type="text/markdown",
            data=f"# {topic}\n\n{outcome.text}".encode(),
        )
        return document.id

    async def _run_job(self, job_id: uuid.UUID) -> None:
        try:
            async with self._database.session() as session:
                job = await session.get(GenerationJob, job_id)
                assert job is not None
                deck = await session.get(Deck, job.deck_id)
                assert deck is not None
                session.expunge_all()
            await self._update_job(job_id, status=JobStatus.RUNNING.value, stage="Starting")

            document_ids = [uuid.UUID(d) for d in job.request.get("document_ids") or []]
            if job.request.get("topic"):
                await self._update_job(job_id, stage="Researching the topic", progress=0.05)
                document_ids.append(await self._topic_note(job, deck))

            async with self._database.session() as session:
                rows = await session.execute(
                    select(DocumentChunk, Document.filename)
                    .join(Document, Document.id == DocumentChunk.document_id)
                    .where(
                        DocumentChunk.user_id == job.user_id,
                        DocumentChunk.document_id.in_(document_ids),
                    )
                    .order_by(
                        Document.created_at, DocumentChunk.document_id, DocumentChunk.chunk_index
                    )
                )
                chunks = [
                    SourceChunk(c.document_id, filename, c.chunk_index, c.content)
                    for c, filename in rows.all()
                ]
                existing = (
                    await session.scalars(select(Card.prompt).where(Card.deck_id == deck.id))
                ).all()
            passages = build_passages(
                chunks,
                max_chars=self._settings.coach_passage_chars,
                max_passages=self._settings.coach_max_passages,
            )
            if not passages:
                raise Invalid("The chosen documents have no text to make cards from")

            await self._update_job(job_id, stage="Writing and checking questions", progress=0.1)
            spent_before = self._spent()

            async def progress(fraction: float) -> None:
                await self._update_job(job_id, progress=round(0.1 + 0.8 * fraction, 3))

            generator = CardGenerator(
                writer=self._models.writer,
                verifier=self._models.verifier,
                min_confidence=self._settings.coach_min_verifier_confidence,
            )
            result = await generator.generate(
                passages,
                DeckSpec(deck.name, Track(deck.track), DeckLanguage(deck.language)),
                max_cards=int(job.request.get("max_cards") or 20),
                focus=job.request.get("focus"),
                existing_prompts=existing,
                on_progress=progress,
            )

            async with self._database.session() as session:
                concepts: dict[str, Concept] = {
                    c.name.lower(): c
                    for c in (
                        await session.scalars(select(Concept).where(Concept.deck_id == deck.id))
                    ).all()
                }
                now = datetime.now(tz=UTC)
                for draft in result.cards:
                    concept = concepts.get(draft.concept.lower())
                    if concept is None:
                        concept = Concept(
                            deck_id=deck.id,
                            user_id=deck.user_id,
                            name=draft.concept,
                            summary=draft.concept_summary,
                        )
                        session.add(concept)
                        await session.flush()
                        concepts[draft.concept.lower()] = concept
                    card = Card(
                        deck_id=deck.id,
                        concept_id=concept.id,
                        user_id=deck.user_id,
                        kind=draft.kind.value,
                        prompt=draft.prompt,
                        choices=draft.choices,
                        answer_index=draft.answer_index,
                        answer=draft.answer,
                        accepted=draft.accepted,
                        explanation=draft.explanation,
                        hints=draft.hints,
                        source_document_id=draft.passage.document_id,
                        source_filename=draft.passage.filename,
                        source_excerpt=draft.excerpt,
                        verification=draft.verification,
                        status=CardStatus.ACTIVE.value,
                        job_id=job_id,
                    )
                    session.add(card)
                    await session.flush()
                    state, due = memory.new_memory(now)
                    session.add(
                        CardState(
                            card_id=card.id,
                            user_id=deck.user_id,
                            deck_id=deck.id,
                            fsrs=state,
                            due=due,
                        )
                    )

            stats = result.stats() | {"cost_usd": round(self._spent() - spent_before, 4)}
            if not result.cards and result.errors:
                await self._update_job(
                    job_id,
                    status=JobStatus.FAILED.value,
                    stage="Failed",
                    error=result.errors[0],
                    stats=stats,
                    finished_at=datetime.now(tz=UTC),
                )
                return
            await self._update_job(
                job_id,
                status=JobStatus.DONE.value,
                stage=f"Made {len(result.cards)} cards",
                progress=1.0,
                stats=stats,
                finished_at=datetime.now(tz=UTC),
            )
        except asyncio.CancelledError:
            await self._update_job(
                job_id, status=JobStatus.FAILED.value, stage="Stopped", error="Stopped"
            )
            raise
        except Exception as exc:  # noqa: BLE001 - a job must always end in a final state
            logger.exception("generation_job_failed", job_id=str(job_id))
            await self._update_job(
                job_id,
                status=JobStatus.FAILED.value,
                stage="Failed",
                error=str(exc)[:500],
                finished_at=datetime.now(tz=UTC),
            )

    def _spent(self) -> float:
        """Credit used so far by the writer and verifier clients (USD)."""
        models = (self._models.writer, self._models.verifier)
        return sum(float(getattr(m, "cost_usd", 0.0) or 0.0) for m in models)

    # ------------------------------------------------------------------ #
    # cards
    # ------------------------------------------------------------------ #
    async def _card(self, session: AsyncSession, card_id: uuid.UUID, user_id: uuid.UUID) -> Card:
        card = await session.get(Card, card_id)
        if card is None or card.user_id != user_id:
            raise NotFound(f"Card {card_id} was not found")
        return card

    async def list_cards(
        self, deck_id: uuid.UUID, user_id: uuid.UUID, status: CardStatus | None = None
    ) -> Sequence[Card]:
        async with self._database.session() as session:
            await self._deck(session, deck_id, user_id)
            stmt = select(Card).where(Card.deck_id == deck_id).order_by(Card.created_at, Card.id)
            if status is not None:
                stmt = stmt.where(Card.status == status.value)
            return (await session.scalars(stmt)).unique().all()

    async def get_card(self, card_id: uuid.UUID, user_id: uuid.UUID) -> Card:
        async with self._database.session() as session:
            return await self._card(session, card_id, user_id)

    async def update_card(self, card_id: uuid.UUID, user_id: uuid.UUID, **changes: Any) -> Card:
        async with self._database.session() as session:
            card = await self._card(session, card_id, user_id)
            for key, value in changes.items():
                if value is not None:
                    setattr(card, key, value.value if hasattr(value, "value") else value)
            if card.kind == CardKind.MCQ.value:
                choices = card.choices or []
                if card.answer_index is None or not 0 <= card.answer_index < len(choices):
                    raise Invalid("answer_index must point at one of the choices")
                card.answer = choices[card.answer_index]
            if len(card.hints) > 3:
                raise Invalid("A card has at most 3 hints")
            await session.flush()
            return card

    async def delete_card(self, card_id: uuid.UUID, user_id: uuid.UUID) -> None:
        async with self._database.session() as session:
            card = await self._card(session, card_id, user_id)
            await session.execute(delete(ReviewLog).where(ReviewLog.card_id == card.id))
            await session.delete(card)  # its CardState goes with it (ORM cascade)

    async def flag_card(self, card_id: uuid.UUID, user_id: uuid.UUID, reason: str) -> Card:
        return await self.update_card(
            card_id, user_id, status=CardStatus.FLAGGED, flag_reason=reason.strip()[:500]
        )

    # ------------------------------------------------------------------ #
    # queue
    # ------------------------------------------------------------------ #
    async def _today_counts(
        self, session: AsyncSession, user_id: uuid.UUID, since: datetime
    ) -> dict[uuid.UUID, tuple[int, int]]:
        rows = await session.execute(
            select(ReviewLog.deck_id, ReviewLog.was_new, func.count())
            .where(ReviewLog.user_id == user_id, ReviewLog.reviewed_at >= since)
            .group_by(ReviewLog.deck_id, ReviewLog.was_new)
        )
        counts: dict[uuid.UUID, list[int]] = defaultdict(lambda: [0, 0])
        for deck_id, was_new, count in rows.all():
            counts[deck_id][0 if was_new else 1] += count
        return {k: (v[0], v[1]) for k, v in counts.items()}

    async def queue(
        self,
        *,
        user_id: uuid.UUID,
        deck_id: uuid.UUID | None = None,
        limit: int = 20,
        now: datetime | None = None,
    ) -> list[QueueItem]:
        """Due reviews first, with new cards mixed in (about 1 in 4)."""
        moment = now or datetime.now(tz=UTC)
        async with self._database.session() as session:
            if deck_id is not None:
                decks = [await self._deck(session, deck_id, user_id)]
            else:
                decks = list(
                    (await session.scalars(select(Deck).where(Deck.user_id == user_id))).all()
                )
            done_today = await self._today_counts(session, user_id, self._day_start(moment))
            reviews: list[Card] = []
            fresh: list[Card] = []
            for deck in decks:
                new_done, reviews_done = done_today.get(deck.id, (0, 0))
                base = (
                    select(Card)
                    .join(CardState, CardState.card_id == Card.id)
                    .where(Card.deck_id == deck.id, Card.status == CardStatus.ACTIVE.value)
                )
                review_room = max(0, deck.reviews_per_day - reviews_done)
                if review_room:
                    reviews += (
                        (
                            await session.scalars(
                                base.where(CardState.reps > 0, CardState.due <= moment)
                                .order_by(CardState.due)
                                .limit(min(review_room, limit))
                            )
                        )
                        .unique()
                        .all()
                    )
                new_room = max(0, deck.new_per_day - new_done)
                if new_room:
                    fresh += (
                        (
                            await session.scalars(
                                base.where(CardState.reps == 0)
                                .order_by(Card.created_at, Card.id)
                                .limit(min(new_room, limit))
                            )
                        )
                        .unique()
                        .all()
                    )
        reviews.sort(key=lambda c: c.state.due)
        items: list[QueueItem] = []
        r = n = 0
        while len(items) < limit and (r < len(reviews) or n < len(fresh)):
            take_new = n < len(fresh) and (r >= len(reviews) or (len(items) + 1) % 4 == 0)
            if take_new:
                items.append(QueueItem(fresh[n], True))
                n += 1
            else:
                items.append(QueueItem(reviews[r], False))
                r += 1
        return items

    # ------------------------------------------------------------------ #
    # review
    # ------------------------------------------------------------------ #
    async def hint(self, card_id: uuid.UUID, user_id: uuid.UUID, level: int) -> str:
        card = await self.get_card(card_id, user_id)
        if not card.hints:
            raise NotFound("This card has no hints")
        return card.hints[min(max(level, 1), len(card.hints)) - 1]

    async def review(
        self,
        card_id: uuid.UUID,
        user_id: uuid.UUID,
        submission: Submission,
        *,
        now: datetime | None = None,
    ) -> ReviewResult:
        moment = now or datetime.now(tz=UTC)
        async with self._database.session() as session:
            card = await self._card(session, card_id, user_id)
            if card.status != CardStatus.ACTIVE.value:
                raise Invalid("This card is paused or flagged; restore it to review it")
            if (
                card.kind == CardKind.MCQ.value
                and submission.choice_index is None
                and not submission.revealed
            ):
                raise Invalid("Pick one of the choices")
            deck = await session.get(Deck, card.deck_id)
            assert deck is not None
            state = card.state
            result = await grade(card, submission, self._models.grader)
            rating = rating_for(card, result, submission)
            was_new = state.reps == 0
            was_review = memory.state_name(state.fsrs) == "review"
            update = memory.review(
                state.fsrs,
                rating,
                desired_retention=deck.desired_retention,
                now=moment,
                duration_ms=submission.duration_ms,
            )
            state.fsrs = update.fsrs
            state.due = update.due
            state.reps += 1
            state.last_review = moment
            if rating is ReviewRating.AGAIN and was_review:
                state.lapses += 1
            session.add(
                ReviewLog(
                    card_id=card.id,
                    user_id=user_id,
                    deck_id=card.deck_id,
                    rating=rating.value,
                    correct=result.correct,
                    partial=result.partial,
                    hints_used=max(0, submission.hints_used),
                    revealed=submission.revealed,
                    duration_ms=submission.duration_ms,
                    response=(
                        submission.response[:2000]
                        if submission.response
                        else (
                            str(submission.choice_index)
                            if submission.choice_index is not None
                            else None
                        )
                    ),
                    grader=result.grader,
                    was_new=was_new,
                    interval_days=update.interval_days,
                    reviewed_at=moment,
                )
            )
            await session.flush()
            return ReviewResult(
                card, result, rating, update.due, update.interval_days, update.state
            )

    # ------------------------------------------------------------------ #
    # stats
    # ------------------------------------------------------------------ #
    async def streak(self, user_id: uuid.UUID, *, now: datetime | None = None) -> int:
        moment = now or datetime.now(tz=UTC)
        async with self._database.session() as session:
            stamps = (
                await session.scalars(
                    select(ReviewLog.reviewed_at)
                    .where(
                        ReviewLog.user_id == user_id,
                        ReviewLog.reviewed_at >= moment - timedelta(days=400),
                    )
                    .order_by(ReviewLog.reviewed_at.desc())
                )
            ).all()
        days = {self._local_date(s if s.tzinfo else s.replace(tzinfo=UTC)) for s in stamps}
        day = self._local_date(moment)
        if day not in days:
            day -= timedelta(days=1)  # today not done yet; the streak is still alive
        count = 0
        while day in days:
            count += 1
            day -= timedelta(days=1)
        return count

    async def deck_stats(
        self, user_id: uuid.UUID, deck_id: uuid.UUID | None = None, *, now: datetime | None = None
    ) -> list[DeckStats]:
        moment = now or datetime.now(tz=UTC)
        since = self._day_start(moment)
        async with self._database.session() as session:
            if deck_id is not None:
                decks = [await self._deck(session, deck_id, user_id)]
            else:
                decks = list(
                    (
                        await session.scalars(
                            select(Deck).where(Deck.user_id == user_id).order_by(Deck.created_at)
                        )
                    ).all()
                )
            done_today = await self._today_counts(session, user_id, since)
            out: list[DeckStats] = []
            for deck in decks:
                stats = DeckStats(deck=deck)
                rows = (
                    (
                        await session.execute(
                            select(Card, CardState)
                            .join(CardState, CardState.card_id == Card.id)
                            .where(Card.deck_id == deck.id)
                        )
                    )
                    .unique()
                    .all()
                )
                per_concept: dict[str, list[float | None]] = defaultdict(list)
                for card, state in rows:
                    stats.total += 1
                    if card.status != CardStatus.ACTIVE.value:
                        stats.paused += 1
                        continue
                    if state.reps == 0:
                        stats.new += 1
                    elif memory.state_name(state.fsrs) == "review":
                        stats.review += 1
                    else:
                        stats.learning += 1
                    if state.reps > 0 and _aware(state.due) <= moment:
                        stats.due_now += 1
                    name = card.concept.name if card.concept else "General"
                    per_concept[name].append(
                        memory.retrievability(
                            state.fsrs, desired_retention=deck.desired_retention, now=moment
                        )
                    )
                new_done, reviews_done = done_today.get(deck.id, (0, 0))
                stats.reviewed_today = new_done + reviews_done
                stats.new_available_today = min(stats.new, max(0, deck.new_per_day - new_done))
                accuracy = (
                    await session.execute(
                        select(func.count(), func.sum(func.cast(ReviewLog.correct, _Int()))).where(
                            and_(
                                ReviewLog.deck_id == deck.id,
                                ReviewLog.was_new.is_(False),
                                ReviewLog.reviewed_at >= moment - timedelta(days=30),
                            )
                        )
                    )
                ).one()
                total, correct = accuracy
                stats.accuracy_30d = round((correct or 0) / total, 3) if total else None
                concepts = []
                for name, values in per_concept.items():
                    seen = [v for v in values if v is not None]
                    concepts.append(
                        {
                            "name": name,
                            "cards": len(values),
                            "studied": len(seen),
                            "strength": round(sum(seen) / len(seen), 3) if seen else None,
                        }
                    )
                concepts.sort(key=lambda c: (c["strength"] is None, c["strength"] or 0))
                stats.concepts = concepts
                out.append(stats)
            return out


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _Int() -> Any:  # noqa: N802 - tiny helper for the boolean-sum cast
    from sqlalchemy import Integer

    return Integer()
