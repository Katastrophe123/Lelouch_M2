"""Build the coach's model clients from settings."""

from __future__ import annotations

from app.coach.fake import FakeCoachLLM
from app.coach.service import CoachModels
from app.config import Settings
from app.llm.claude import ClaudeAgentLLMClient


def build_coach_models(settings: Settings) -> CoachModels:
    if settings.coach_backend == "fake":
        return CoachModels(
            writer=FakeCoachLLM(),
            verifier=FakeCoachLLM("fake-verifier"),
            grader=FakeCoachLLM("fake-grader"),
        )

    def client(model: str) -> ClaudeAgentLLMClient:
        return ClaudeAgentLLMClient(
            model=model,
            timeout_seconds=settings.coach_llm_timeout_seconds,
            workspace=f"{settings.agent_workspace_dir}/_coach",
            cli_path=settings.agent_cli_path,
        )

    return CoachModels(
        writer=client(settings.coach_generation_model),
        verifier=client(settings.coach_verifier_model),
        grader=client(settings.coach_grader_model),
    )
