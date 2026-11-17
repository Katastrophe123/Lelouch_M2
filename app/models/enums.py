"""Domain enumerations shared by the graph, the database and the API."""

from __future__ import annotations

from enum import StrEnum


class Subject(StrEnum):
    """Academic subject inferred by the ``understand`` node."""

    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    COMPUTER_SCIENCE = "computer_science"
    ECONOMICS = "economics"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    ENGLISH = "english"
    OTHER = "other"


class Difficulty(StrEnum):
    INTRODUCTORY = "introductory"
    INTERMEDIATE = "advanced_school"
    UNIVERSITY = "university"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AnswerStatus(StrEnum):
    """Terminal state of one tutoring turn."""

    PENDING = "pending"
    #: Agent-mode turn that finished normally. Deliberately not "verified":
    #: agent answers are not passed through the tutor's self-check graph.
    COMPLETED = "completed"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"


class ConversationMode(StrEnum):
    """Which engine answers a conversation."""

    #: The original LangGraph tutor: understand -> solve -> self-check -> explain.
    TUTOR = "tutor"
    #: The Claude Agent SDK agent with tools, MCP servers and your documents.
    AGENT = "agent"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class StepStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class NodeName(StrEnum):
    """Graph node identifiers, also used as step labels in the trace."""

    UNDERSTAND = "understand"
    SOLVE = "solve"
    SELF_CHECK = "self_check"
    EXPLAIN = "explain"
    FALLBACK = "fallback"

    # Agent-mode step kinds. The specific tool lives in the step's output.
    AGENT_START = "agent_start"
    THINKING = "thinking"
    TOOL = "tool"
    ANSWER = "answer"


class RoutingDecision(StrEnum):
    """Outcome of a conditional edge, persisted for auditability."""

    TO_SOLVE = "to_solve"
    TO_SELF_CHECK = "to_self_check"
    TO_EXPLAIN = "to_explain"
    RETRY_SOLVE = "retry_solve"
    TO_FALLBACK_CHECK_FAILED = "to_fallback_check_failed"
    TO_FALLBACK_NODE_ERROR = "to_fallback_node_error"
    TO_FALLBACK_STEP_LIMIT = "to_fallback_step_limit"
    TO_FALLBACK_NOT_ANSWERABLE = "to_fallback_not_answerable"
    END = "end"


class FallbackReason(StrEnum):
    """Why a turn returned an unverified answer instead of a checked one."""

    SELF_CHECK_FAILED = "self_check_failed"
    NODE_ERROR = "node_error"
    STEP_LIMIT_REACHED = "step_limit_reached"
    NOT_ANSWERABLE = "not_answerable"


class StreamEventType(StrEnum):
    """Server-Sent Event names emitted by the streaming endpoint."""

    STEP = "step"
    TOKEN = "token"
    DONE = "done"
    ERROR = "error"


# --------------------------------------------------------------------------- #
# Memory coach
# --------------------------------------------------------------------------- #
class Track(StrEnum):
    """Shapes how questions are written for a deck."""

    #: UPSC Civil Services: Prelims-style statement MCQs, facts, concepts.
    UPSC = "upsc"
    #: AI engineering: AI, ML, DL, RL, RL environments and evals.
    TECH = "tech"
    GENERAL = "general"


class DeckLanguage(StrEnum):
    ENGLISH = "en"
    HINDI = "hi"
    #: Hindi written in Latin script, mixed with English terms.
    HINGLISH = "hinglish"


class CardKind(StrEnum):
    MCQ = "mcq"
    CLOZE = "cloze"
    NUMERIC = "numeric"
    SHORT = "short"


class CardStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    #: Reported by the learner as wrong or unclear; kept out of reviews.
    FLAGGED = "flagged"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ReviewRating(StrEnum):
    """FSRS ratings, by name. Values match ``fsrs.Rating``."""

    AGAIN = "again"
    HARD = "hard"
    GOOD = "good"
    EASY = "easy"
