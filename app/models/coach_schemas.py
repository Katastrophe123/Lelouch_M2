"""Request and response shapes for the memory coach API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import (
    CardKind,
    CardStatus,
    DeckLanguage,
    JobStatus,
    ReviewRating,
    Track,
)


class PresetResponse(BaseModel):
    track: Track
    name: str
    description: str


class DeckCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    track: Track = Track.GENERAL
    language: DeckLanguage = DeckLanguage.ENGLISH
    description: Annotated[str | None, Field(max_length=2000)] = None
    exam_date: datetime | None = None
    new_per_day: Annotated[int, Field(ge=0, le=200)] = 20
    desired_retention: Annotated[float, Field(ge=0.7, le=0.97)] = 0.9


class DeckUpdate(BaseModel):
    name: Annotated[str | None, Field(min_length=1, max_length=120)] = None
    language: DeckLanguage | None = None
    description: Annotated[str | None, Field(max_length=2000)] = None
    exam_date: datetime | None = None
    new_per_day: Annotated[int | None, Field(ge=0, le=200)] = None
    reviews_per_day: Annotated[int | None, Field(ge=1, le=2000)] = None
    desired_retention: Annotated[float | None, Field(ge=0.7, le=0.97)] = None


class DeckResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    track: Track
    language: DeckLanguage
    description: str | None = None
    exam_date: datetime | None = None
    new_per_day: int
    reviews_per_day: int
    desired_retention: float
    created_at: datetime


class GenerateRequest(BaseModel):
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    #: Research a topic, save the note to your documents, then make cards from it.
    topic: Annotated[str | None, Field(max_length=300)] = None
    #: Optional emphasis, e.g. "fundamental rights and their exceptions".
    focus: Annotated[str | None, Field(max_length=300)] = None
    max_cards: Annotated[int | None, Field(ge=1, le=200)] = None


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    deck_id: uuid.UUID
    status: JobStatus
    stage: str | None = None
    progress: float
    stats: dict[str, Any]
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class CardResponse(BaseModel):
    """Full card, including the answer: for the library and editing."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    deck_id: uuid.UUID
    concept: str | None = None
    kind: CardKind
    prompt: str
    choices: list[str] | None = None
    answer_index: int | None = None
    answer: str
    accepted: list[str] | None = None
    explanation: str | None = None
    hints: list[str]
    source_filename: str | None = None
    source_excerpt: str | None = None
    verification: dict[str, Any] | None = None
    status: CardStatus
    flag_reason: str | None = None
    due: datetime | None = None
    reps: int = 0
    lapses: int = 0
    created_at: datetime


class CardUpdate(BaseModel):
    prompt: Annotated[str | None, Field(min_length=3, max_length=4000)] = None
    choices: list[str] | None = None
    answer_index: int | None = None
    answer: Annotated[str | None, Field(max_length=2000)] = None
    accepted: list[str] | None = None
    explanation: Annotated[str | None, Field(max_length=4000)] = None
    hints: list[str] | None = None
    status: CardStatus | None = None


class FlagRequest(BaseModel):
    reason: Annotated[str, Field(min_length=1, max_length=500)]


class QueueCard(BaseModel):
    """A card as shown before answering: no answer, no explanation."""

    id: uuid.UUID
    deck_id: uuid.UUID
    deck_name: str
    concept: str | None = None
    kind: CardKind
    prompt: str
    choices: list[str] | None = None
    hint_count: int
    is_new: bool


class QueueResponse(BaseModel):
    cards: list[QueueCard]
    due_total: int
    new_total: int


class HintResponse(BaseModel):
    level: int
    hint: str
    remaining: int


class ReviewSubmit(BaseModel):
    response: Annotated[str | None, Field(max_length=4000)] = None
    choice_index: Annotated[int | None, Field(ge=0, le=9)] = None
    hints_used: Annotated[int, Field(ge=0, le=3)] = 0
    duration_ms: Annotated[int | None, Field(ge=0, le=3_600_000)] = None
    revealed: bool = False


class ReviewResponse(BaseModel):
    correct: bool
    partial: bool
    rating: ReviewRating
    feedback: str
    grader: str
    answer: str
    answer_index: int | None = None
    explanation: str | None = None
    source_filename: str | None = None
    source_excerpt: str | None = None
    next_due: datetime
    interval_days: float
    state: str


class ConceptStrength(BaseModel):
    name: str
    cards: int
    studied: int
    strength: float | None = Field(default=None, description="Average recall probability now")


class DeckStatsResponse(BaseModel):
    deck: DeckResponse
    total: int
    new: int
    learning: int
    review: int
    paused: int
    due_now: int
    new_available_today: int
    reviewed_today: int
    accuracy_30d: float | None = None
    concepts: list[ConceptStrength]


class StatsResponse(BaseModel):
    streak_days: int
    due_now: int
    new_available_today: int
    reviewed_today: int
    decks: list[DeckStatsResponse]
