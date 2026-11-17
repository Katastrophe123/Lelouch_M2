"""memory coach: decks, concepts, cards, FSRS state, reviews, generation jobs

Revision ID: c41f8e2a9d10
Revises: 7d2a9c41e5b3
Create Date: 2026-11-17 13:30:00+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c41f8e2a9d10"
down_revision: str | None = "7d2a9c41e5b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "decks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("track", sa.String(16), nullable=False),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("desired_retention", sa.Float(), nullable=False),
        sa.Column("new_per_day", sa.Integer(), nullable=False),
        sa.Column("reviews_per_day", sa.Integer(), nullable=False),
        sa.Column("exam_date", TS, nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
        sa.UniqueConstraint("user_id", "name", name="uq_decks_user_name"),
    )
    op.create_index("ix_decks_user_id", "decks", ["user_id"])

    op.create_table(
        "concepts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "deck_id", sa.Uuid(), sa.ForeignKey("decks.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.UniqueConstraint("deck_id", "name", name="uq_concepts_deck_name"),
    )
    op.create_index("ix_concepts_deck_id", "concepts", ["deck_id"])
    op.create_index("ix_concepts_user_id", "concepts", ["user_id"])

    op.create_table(
        "cards",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "deck_id", sa.Uuid(), sa.ForeignKey("decks.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "concept_id",
            sa.Uuid(),
            sa.ForeignKey("concepts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("choices", sa.JSON(), nullable=True),
        sa.Column("answer_index", sa.Integer(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("accepted", sa.JSON(), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("hints", sa.JSON(), nullable=False),
        sa.Column(
            "source_document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_filename", sa.String(255), nullable=True),
        sa.Column("source_excerpt", sa.Text(), nullable=True),
        sa.Column("verification", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("flag_reason", sa.Text(), nullable=True),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", TS, nullable=False),
    )
    for column in ("deck_id", "concept_id", "user_id"):
        op.create_index(f"ix_cards_{column}", "cards", [column])

    op.create_table(
        "card_states",
        sa.Column(
            "card_id", sa.Uuid(), sa.ForeignKey("cards.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("deck_id", sa.Uuid(), nullable=False),
        sa.Column("fsrs", sa.JSON(), nullable=False),
        sa.Column("due", TS, nullable=False),
        sa.Column("reps", sa.Integer(), nullable=False),
        sa.Column("lapses", sa.Integer(), nullable=False),
        sa.Column("last_review", TS, nullable=True),
    )
    for column in ("user_id", "deck_id", "due"):
        op.create_index(f"ix_card_states_{column}", "card_states", [column])

    op.create_table(
        "review_logs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "card_id", sa.Uuid(), sa.ForeignKey("cards.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("deck_id", sa.Uuid(), nullable=False),
        sa.Column("rating", sa.String(8), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("partial", sa.Boolean(), nullable=False),
        sa.Column("hints_used", sa.Integer(), nullable=False),
        sa.Column("revealed", sa.Boolean(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("response", sa.Text(), nullable=True),
        sa.Column("grader", sa.String(16), nullable=False),
        sa.Column("was_new", sa.Boolean(), nullable=False),
        sa.Column("interval_days", sa.Float(), nullable=True),
        sa.Column("reviewed_at", TS, nullable=False),
    )
    for column in ("card_id", "user_id", "deck_id", "reviewed_at"):
        op.create_index(f"ix_review_logs_{column}", "review_logs", [column])

    op.create_table(
        "generation_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "deck_id", sa.Uuid(), sa.ForeignKey("decks.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("stage", sa.String(120), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("stats", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("finished_at", TS, nullable=True),
    )
    op.create_index("ix_generation_jobs_deck_id", "generation_jobs", ["deck_id"])
    op.create_index("ix_generation_jobs_user_id", "generation_jobs", ["user_id"])


def downgrade() -> None:
    for table in ("generation_jobs", "review_logs", "card_states", "cards", "concepts", "decks"):
        op.drop_table(table)
