"""Memory coach API: decks, generation, cards, the daily review, progress."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Path, Query, Response, status

from app.api.deps import CoachServiceDep, CurrentUserDep
from app.api.errors import BadRequestError, NotFoundError
from app.coach.grading import Submission
from app.coach.presets import PRESETS
from app.coach.service import Invalid, NotFound
from app.db.models import Card
from app.models.coach_schemas import (
    CardResponse,
    CardUpdate,
    ConceptStrength,
    DeckCreate,
    DeckResponse,
    DeckStatsResponse,
    DeckUpdate,
    FlagRequest,
    GenerateRequest,
    HintResponse,
    JobResponse,
    PresetResponse,
    QueueCard,
    QueueResponse,
    ReviewResponse,
    ReviewSubmit,
    StatsResponse,
)
from app.models.enums import CardStatus

router = APIRouter(tags=["coach"])

DeckId = Annotated[uuid.UUID, Path(description="Deck identifier")]
CardId = Annotated[uuid.UUID, Path(description="Card identifier")]


def _translate(exc: Exception) -> Exception:
    if isinstance(exc, NotFound):
        return NotFoundError(str(exc))
    if isinstance(exc, Invalid):
        return BadRequestError(str(exc))
    return exc


def card_response(card: Card) -> CardResponse:
    fields = {
        name: getattr(card, name) for name in CardResponse.model_fields if hasattr(card, name)
    }
    fields.update(
        concept=card.concept.name if card.concept else None,
        due=card.state.due if card.state else None,
        reps=card.state.reps if card.state else 0,
        lapses=card.state.lapses if card.state else 0,
    )
    return CardResponse.model_validate(fields)


# --- decks ------------------------------------------------------------------
@router.get(
    "/decks/presets", response_model=list[PresetResponse], summary="Starter decks per track"
)
async def presets(_user: CurrentUserDep) -> list[PresetResponse]:
    return [PresetResponse(track=p.track, name=p.name, description=p.description) for p in PRESETS]


@router.get("/decks", response_model=list[DeckResponse], summary="Your decks")
async def list_decks(user: CurrentUserDep, coach: CoachServiceDep) -> list[DeckResponse]:
    return [DeckResponse.model_validate(d) for d in await coach.list_decks(user.id)]


@router.post(
    "/decks",
    response_model=DeckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a deck",
)
async def create_deck(
    payload: DeckCreate, user: CurrentUserDep, coach: CoachServiceDep
) -> DeckResponse:
    try:
        deck = await coach.create_deck(user_id=user.id, **payload.model_dump())
    except Exception as exc:
        raise _translate(exc) from exc
    return DeckResponse.model_validate(deck)


@router.get("/decks/{deck_id}", response_model=DeckResponse, summary="One deck")
async def get_deck(deck_id: DeckId, user: CurrentUserDep, coach: CoachServiceDep) -> DeckResponse:
    try:
        return DeckResponse.model_validate(await coach.get_deck(deck_id, user.id))
    except Exception as exc:
        raise _translate(exc) from exc


@router.patch("/decks/{deck_id}", response_model=DeckResponse, summary="Change deck settings")
async def update_deck(
    deck_id: DeckId, payload: DeckUpdate, user: CurrentUserDep, coach: CoachServiceDep
) -> DeckResponse:
    try:
        deck = await coach.update_deck(deck_id, user.id, **payload.model_dump(exclude_unset=True))
    except Exception as exc:
        raise _translate(exc) from exc
    return DeckResponse.model_validate(deck)


@router.delete(
    "/decks/{deck_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a deck and its cards",
)
async def delete_deck(deck_id: DeckId, user: CurrentUserDep, coach: CoachServiceDep) -> Response:
    try:
        await coach.delete_deck(deck_id, user.id)
    except Exception as exc:
        raise _translate(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- generation ---------------------------------------------------------------
@router.post(
    "/decks/{deck_id}/generate",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Make verified cards from documents or a topic (runs in the background)",
)
async def generate(
    deck_id: DeckId, payload: GenerateRequest, user: CurrentUserDep, coach: CoachServiceDep
) -> JobResponse:
    try:
        job = await coach.start_generation(deck_id=deck_id, user_id=user.id, **payload.model_dump())
    except Exception as exc:
        raise _translate(exc) from exc
    return JobResponse.model_validate(job)


@router.get("/jobs/{job_id}", response_model=JobResponse, summary="Progress of a generation job")
async def get_job(
    job_id: Annotated[uuid.UUID, Path()], user: CurrentUserDep, coach: CoachServiceDep
) -> JobResponse:
    try:
        return JobResponse.model_validate(await coach.get_job(job_id, user.id))
    except Exception as exc:
        raise _translate(exc) from exc


# --- cards --------------------------------------------------------------------
@router.get("/decks/{deck_id}/cards", response_model=list[CardResponse], summary="Cards in a deck")
async def list_cards(
    deck_id: DeckId,
    user: CurrentUserDep,
    coach: CoachServiceDep,
    card_status: Annotated[CardStatus | None, Query(alias="status")] = None,
) -> list[CardResponse]:
    try:
        cards = await coach.list_cards(deck_id, user.id, card_status)
    except Exception as exc:
        raise _translate(exc) from exc
    return [card_response(c) for c in cards]


@router.get("/cards/{card_id}", response_model=CardResponse, summary="One card")
async def get_card(card_id: CardId, user: CurrentUserDep, coach: CoachServiceDep) -> CardResponse:
    try:
        return card_response(await coach.get_card(card_id, user.id))
    except Exception as exc:
        raise _translate(exc) from exc


@router.patch(
    "/cards/{card_id}", response_model=CardResponse, summary="Edit, pause or restore a card"
)
async def update_card(
    card_id: CardId, payload: CardUpdate, user: CurrentUserDep, coach: CoachServiceDep
) -> CardResponse:
    try:
        card = await coach.update_card(card_id, user.id, **payload.model_dump(exclude_unset=True))
        return card_response(await coach.get_card(card.id, user.id))
    except Exception as exc:
        raise _translate(exc) from exc


@router.post(
    "/cards/{card_id}/flag", response_model=CardResponse, summary="Report a wrong or unclear card"
)
async def flag_card(
    card_id: CardId, payload: FlagRequest, user: CurrentUserDep, coach: CoachServiceDep
) -> CardResponse:
    try:
        await coach.flag_card(card_id, user.id, payload.reason)
        return card_response(await coach.get_card(card_id, user.id))
    except Exception as exc:
        raise _translate(exc) from exc


@router.delete("/cards/{card_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a card")
async def delete_card(card_id: CardId, user: CurrentUserDep, coach: CoachServiceDep) -> Response:
    try:
        await coach.delete_card(card_id, user.id)
    except Exception as exc:
        raise _translate(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- review -------------------------------------------------------------------
@router.get("/review/queue", response_model=QueueResponse, summary="What to review now")
async def review_queue(
    user: CurrentUserDep,
    coach: CoachServiceDep,
    deck_id: Annotated[uuid.UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> QueueResponse:
    try:
        items = await coach.queue(user_id=user.id, deck_id=deck_id, limit=limit)
        decks = {d.id: d.name for d in await coach.list_decks(user.id)}
    except Exception as exc:
        raise _translate(exc) from exc
    return QueueResponse(
        cards=[
            QueueCard(
                id=item.card.id,
                deck_id=item.card.deck_id,
                deck_name=decks.get(item.card.deck_id, ""),
                concept=item.card.concept.name if item.card.concept else None,
                kind=item.card.kind,  # type: ignore[arg-type]
                prompt=item.card.prompt,
                choices=item.card.choices,
                hint_count=len(item.card.hints),
                is_new=item.is_new,
            )
            for item in items
        ],
        due_total=sum(not i.is_new for i in items),
        new_total=sum(i.is_new for i in items),
    )


@router.get("/cards/{card_id}/hint", response_model=HintResponse, summary="The next hint (1 to 3)")
async def hint(
    card_id: CardId,
    user: CurrentUserDep,
    coach: CoachServiceDep,
    level: Annotated[int, Query(ge=1, le=3)] = 1,
) -> HintResponse:
    try:
        card = await coach.get_card(card_id, user.id)
        text = await coach.hint(card_id, user.id, level)
    except Exception as exc:
        raise _translate(exc) from exc
    return HintResponse(level=level, hint=text, remaining=max(0, len(card.hints) - level))


@router.post("/cards/{card_id}/review", response_model=ReviewResponse, summary="Answer a card")
async def review(
    card_id: CardId, payload: ReviewSubmit, user: CurrentUserDep, coach: CoachServiceDep
) -> ReviewResponse:
    try:
        result = await coach.review(card_id, user.id, Submission(**payload.model_dump()))
    except Exception as exc:
        raise _translate(exc) from exc
    card = result.card
    return ReviewResponse(
        correct=result.grade.correct,
        partial=result.grade.partial,
        rating=result.rating,
        feedback=result.grade.feedback,
        grader=result.grade.grader,
        answer=card.answer,
        answer_index=card.answer_index,
        explanation=card.explanation,
        source_filename=card.source_filename,
        source_excerpt=card.source_excerpt,
        next_due=result.next_due,
        interval_days=result.interval_days,
        state=result.state,
    )


# --- progress -----------------------------------------------------------------
@router.get(
    "/stats", response_model=StatsResponse, summary="Streak, due counts and concept strength"
)
async def stats(
    user: CurrentUserDep,
    coach: CoachServiceDep,
    deck_id: Annotated[uuid.UUID | None, Query()] = None,
) -> StatsResponse:
    try:
        decks = await coach.deck_stats(user.id, deck_id)
        streak = await coach.streak(user.id)
    except Exception as exc:
        raise _translate(exc) from exc
    items = [
        DeckStatsResponse(
            deck=DeckResponse.model_validate(s.deck),
            total=s.total,
            new=s.new,
            learning=s.learning,
            review=s.review,
            paused=s.paused,
            due_now=s.due_now,
            new_available_today=s.new_available_today,
            reviewed_today=s.reviewed_today,
            accuracy_30d=s.accuracy_30d,
            concepts=[ConceptStrength(**c) for c in s.concepts],
        )
        for s in decks
    ]
    return StatsResponse(
        streak_days=streak,
        due_now=sum(i.due_now for i in items),
        new_available_today=sum(i.new_available_today for i in items),
        reviewed_today=sum(i.reviewed_today for i in items),
        decks=items,
    )
