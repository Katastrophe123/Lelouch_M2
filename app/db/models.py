"""SQLAlchemy 2.0 ORM models.

This is the structural heart of the v1 -> v2 upgrade: v1 kept the whole
conversation in ``st.session_state``, so it vanished on refresh and could not
be queried. Here users, conversations and messages are first-class rows, and
each assistant message carries the agent trace that produced it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampType, utcnow
from app.models.enums import (
    AnswerStatus,
    CardStatus,
    ConversationMode,
    ConversationStatus,
    DeckLanguage,
    JobStatus,
    MessageRole,
    Track,
)

#: Width of the stored embeddings. Must match ``EMBEDDING_MODEL``; the default
#: BAAI/bge-small-en-v1.5 produces 384-dimensional vectors. Changing the model
#: to one with a different width needs a migration.
EMBEDDING_DIM = 384

#: pgvector on PostgreSQL; plain JSON on SQLite so the tests need no server.
#: ``with_variant`` keeps pgvector's comparator, so ``cosine_distance`` works.
EmbeddingType = Vector(EMBEDDING_DIM).with_variant(JSON(), "sqlite")


class User(Base):
    """A learner. ``username`` is the identity carried in the JWT subject."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User id={self.id} username={self.username}>"


class Conversation(Base):
    """A tutoring thread owned by one user."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New conversation")
    subject: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ConversationStatus.ACTIVE, index=True
    )
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ConversationMode.TUTOR, server_default="tutor"
    )
    # Agent mode only: the Claude model for this thread, and the Agent SDK
    # session id that lets the next turn resume with full context.
    agent_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at, Message.sequence",
        lazy="selectin",
    )

    __table_args__ = (
        # Serves `GET /conversations` for one user, ordered newest-first.
        Index("ix_conversations_user_id_updated_at", "user_id", "updated_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Conversation id={self.id} user_id={self.user_id}>"


class Message(Base):
    """One message in a conversation.

    Assistant messages additionally store the outcome of the graph run:
    status, fallback reason, confidence, retry count, latency and the full
    node-by-node ``trace``.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # --- assistant-only run metadata -------------------------------------
    answer_status: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    fallback_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    retry_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # The node-by-node trace, stored as JSONB on PostgreSQL. Kept on the
    # message rather than in a separate table so the schema stays the three
    # entities the API exposes; see README "Design decisions".
    trace: Mapped[list[dict[str, Any]]] = mapped_column(
        Base.json_type, nullable=False, default=list
    )

    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        UniqueConstraint("conversation_id", "sequence", name="uq_messages_conversation_sequence"),
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )

    @property
    def is_assistant(self) -> bool:
        return self.role == MessageRole.ASSISTANT.value

    @property
    def is_verified(self) -> bool:
        return self.answer_status == AnswerStatus.VERIFIED.value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Message id={self.id} role={self.role} seq={self.sequence}>"


class Document(Base):
    """A file the user uploaded to their knowledge base."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)

    user: Mapped[User] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Document id={self.id} filename={self.filename}>"


class DocumentChunk(Base):
    """One embedded passage of a document, the unit of retrieval."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Denormalised from the document so search is one indexed, user-scoped scan.
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingType, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_position"),
        # Approximate nearest-neighbour index for cosine search (PostgreSQL).
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# --------------------------------------------------------------------------- #
# Memory coach
# --------------------------------------------------------------------------- #
class Deck(Base):
    """A subject the learner is studying, e.g. "Indian Polity" or "RL basics"."""

    __tablename__ = "decks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    track: Mapped[str] = mapped_column(String(16), nullable=False, default=Track.GENERAL)
    language: Mapped[str] = mapped_column(String(16), nullable=False, default=DeckLanguage.ENGLISH)
    #: Share of cards the scheduler aims to have you remember (FSRS).
    desired_retention: Mapped[float] = mapped_column(Float, nullable=False, default=0.9)
    new_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    reviews_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=200)
    exam_date: Mapped[datetime | None] = mapped_column(TimestampType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_decks_user_name"),)


class Concept(Base):
    """One idea within a deck; cards test it from different angles."""

    __tablename__ = "concepts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    deck_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("decks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)

    __table_args__ = (UniqueConstraint("deck_id", "name", name="uq_concepts_deck_name"),)


class Card(Base):
    """A single question, with its answer, hints and the passage it came from."""

    __tablename__ = "cards"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    deck_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("decks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(), ForeignKey("concepts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    choices: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    answer_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    #: Other answers that count as correct for blanks and short answers.
    accepted: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    hints: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    source_document_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    verification: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=CardStatus.ACTIVE)
    flag_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)

    state: Mapped[CardState] = relationship(
        back_populates="card", cascade="all, delete-orphan", uselist=False, lazy="joined"
    )
    concept: Mapped[Concept | None] = relationship(lazy="joined")


class CardState(Base):
    """FSRS memory state for one card. ``due`` is denormalised for queries."""

    __tablename__ = "card_states"

    card_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("cards.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(), nullable=False, index=True)
    deck_id: Mapped[uuid.UUID] = mapped_column(Uuid(), nullable=False, index=True)
    fsrs: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    due: Mapped[datetime] = mapped_column(TimestampType, nullable=False, index=True)
    reps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lapses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_review: Mapped[datetime | None] = mapped_column(TimestampType, nullable=True)

    card: Mapped[Card] = relationship(back_populates="state")


class ReviewLog(Base):
    """Every answer, which is what FSRS and the progress view learn from."""

    __tablename__ = "review_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    card_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("cards.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(), nullable=False, index=True)
    deck_id: Mapped[uuid.UUID] = mapped_column(Uuid(), nullable=False, index=True)
    rating: Mapped[str] = mapped_column(String(8), nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    partial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    hints_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    revealed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    grader: Mapped[str] = mapped_column(String(16), nullable=False)
    was_new: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    interval_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(
        TimestampType, nullable=False, default=utcnow, index=True
    )


class GenerationJob(Base):
    """A background run that turns material into cards."""

    __tablename__ = "generation_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    deck_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("decks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid(), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.PENDING)
    #: What to make cards from: {"document_ids": [...]} or {"topic": "..."}.
    request: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    stage: Mapped[str | None] = mapped_column(String(120), nullable=True)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(TimestampType, nullable=True)
