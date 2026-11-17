"""Starter decks for each track."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.enums import Track


@dataclass(frozen=True, slots=True)
class DeckPreset:
    track: Track
    name: str
    description: str


PRESETS: tuple[DeckPreset, ...] = (
    # --- UPSC Civil Services (GS papers and Prelims) -------------------------
    DeckPreset(Track.UPSC, "Indian Polity & Constitution", "Articles, schedules, amendments, constitutional and statutory bodies, landmark judgments (GS II)."),
    DeckPreset(Track.UPSC, "Modern Indian History", "1757 to 1947: movements, acts, sessions, leaders and organisations (GS I)."),
    DeckPreset(Track.UPSC, "Ancient & Medieval History", "Dynasties, inscriptions, travellers, administration and society (GS I)."),
    DeckPreset(Track.UPSC, "Art & Culture", "Architecture, painting, dance, music, literature, festivals and heritage sites (GS I)."),
    DeckPreset(Track.UPSC, "Geography", "Physical, Indian and world geography, resources and mapping (GS I)."),
    DeckPreset(Track.UPSC, "Indian Economy", "Planning, banking, fiscal and monetary policy, budget and schemes (GS III)."),
    DeckPreset(Track.UPSC, "Environment & Ecology", "Biodiversity, protected areas, conventions, climate change and pollution (GS III)."),
    DeckPreset(Track.UPSC, "Science & Technology", "Space, defence, biotech, IT and emerging technology for Prelims and GS III."),
    DeckPreset(Track.UPSC, "International Relations", "India's neighbourhood, groupings, treaties and global institutions (GS II)."),
    DeckPreset(Track.UPSC, "Current Affairs", "Recent schemes, reports, indices, appointments and events, linked to static topics."),
    DeckPreset(Track.UPSC, "Ethics, Integrity & Aptitude", "Thinkers, terms, case-study frameworks and values (GS IV)."),
    # --- AI engineering --------------------------------------------------------
    DeckPreset(Track.TECH, "ML Fundamentals", "Bias-variance, regularisation, evaluation metrics, classic models and optimisation."),
    DeckPreset(Track.TECH, "Deep Learning", "Backpropagation, architectures, normalisation, initialisation and training dynamics."),
    DeckPreset(Track.TECH, "Transformers & LLMs", "Attention, tokenisation, pre-training, fine-tuning, RLHF, inference and scaling laws."),
    DeckPreset(Track.TECH, "Reinforcement Learning", "MDPs, Bellman equations, value and policy methods, PPO, exploration and reward design."),
    DeckPreset(Track.TECH, "RL Environments & Evals", "Gymnasium-style environments, rollouts, verifiers, benchmarks and harnesses such as Harbor and Terminal-Bench."),
    DeckPreset(Track.TECH, "AI Agents & Tool Use", "Agent loops, tool calling, MCP, planning, memory, sandboxes and evaluation."),
)  # fmt: skip
