"""FSRS memory model, wrapped so the rest of the app never touches its types.

FSRS (Free Spaced Repetition Scheduler; Ye et al., KDD 2022) models each card
by difficulty, stability and retrievability, and schedules the next review for
when recall probability falls to the deck's ``desired_retention`` (90% by
default). State is stored as the library's own dict, so upgrading FSRS or
re-fitting its parameters never needs a schema change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from fsrs import Card as FsrsCard
from fsrs import Rating, Scheduler, State

from app.models.enums import ReviewRating

_RATING = {
    ReviewRating.AGAIN: Rating.Again,
    ReviewRating.HARD: Rating.Hard,
    ReviewRating.GOOD: Rating.Good,
    ReviewRating.EASY: Rating.Easy,
}


@dataclass(frozen=True, slots=True)
class MemoryUpdate:
    fsrs: dict[str, Any]
    due: datetime
    interval_days: float
    state: str


def _scheduler(desired_retention: float, *, fuzz: bool = True) -> Scheduler:
    return Scheduler(desired_retention=desired_retention, enable_fuzzing=fuzz)


def new_memory(now: datetime | None = None) -> tuple[dict[str, Any], datetime]:
    """A fresh card, due immediately."""
    card = FsrsCard(due=now or datetime.now(tz=UTC))
    return dict(card.to_dict()), card.due


def review(
    memory: dict[str, Any],
    rating: ReviewRating,
    *,
    desired_retention: float = 0.9,
    now: datetime | None = None,
    duration_ms: int | None = None,
    fuzz: bool = True,
) -> MemoryUpdate:
    when = now or datetime.now(tz=UTC)
    card = FsrsCard.from_dict(cast(Any, memory))
    updated, _ = _scheduler(desired_retention, fuzz=fuzz).review_card(
        card, _RATING[rating], review_datetime=when, review_duration=duration_ms
    )
    interval = max(0.0, (updated.due - when).total_seconds() / 86_400)
    return MemoryUpdate(
        fsrs=dict(updated.to_dict()),
        due=updated.due,
        interval_days=round(interval, 4),
        state=State(updated.state).name.lower(),
    )


def retrievability(
    memory: dict[str, Any], *, desired_retention: float = 0.9, now: datetime | None = None
) -> float | None:
    """Probability of recalling the card now; ``None`` before its first review."""
    card = FsrsCard.from_dict(cast(Any, memory))
    if card.last_review is None:
        return None
    return float(
        _scheduler(desired_retention).get_card_retrievability(
            card, current_datetime=now or datetime.now(tz=UTC)
        )
    )


def state_name(memory: dict[str, Any]) -> str:
    return State(FsrsCard.from_dict(cast(Any, memory)).state).name.lower()
