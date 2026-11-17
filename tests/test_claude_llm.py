"""ClaudeAgentLLMClient with the engine replaced by a scripted ``query``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import claude_agent_sdk
import pytest

from app.config import Settings
from app.llm.base import LLMPermanentError, LLMTransientError
from app.llm.claude import ClaudeAgentLLMClient
from app.llm.factory import build_llm_client
from tests import agent_fakes as f


def install(
    monkeypatch: pytest.MonkeyPatch, *messages: Any, error: Exception | None = None
) -> list[Any]:
    seen: list[Any] = []

    def fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        seen.append((prompt, options))

        async def stream() -> AsyncIterator[Any]:
            if error is not None:
                raise error
            for message in messages:
                yield message

        return stream()

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    return seen


async def test_runs_with_no_tools_and_tracks_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    seen = install(monkeypatch, f.assistant(f.text('{"ok": true}')), f.result('{"ok": true}'))
    client = ClaudeAgentLLMClient(model="claude-haiku-5-5", workspace=str(tmp_path))
    assert await client.generate_json(system="Grade it", user="2+2=4?") == {"ok": True}
    prompt, options = seen[0]
    assert prompt == "2+2=4?"
    assert options.tools == [] and options.max_turns == 1 and options.system_prompt == "Grade it"
    assert options.model == "claude-haiku-5-5" and options.permission_mode == "dontAsk"
    assert client.cost_usd == pytest.approx(0.0421)


async def test_falls_back_to_assistant_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    install(monkeypatch, f.assistant(f.text("hello")), f.result(""))
    response = await ClaudeAgentLLMClient(model="m", workspace=str(tmp_path)).generate(
        system="s", user="u"
    )
    assert response.text == "hello"


async def test_errors_are_classified(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    client = ClaudeAgentLLMClient(model="m", workspace=str(tmp_path))
    install(monkeypatch, error=RuntimeError("Authentication error: please /login"))
    with pytest.raises(LLMPermanentError, match="not signed in"):
        await client.generate(system="s", user="u")
    install(monkeypatch, error=RuntimeError("connection reset"))
    with pytest.raises(LLMTransientError):
        await client.generate(system="s", user="u")
    install(monkeypatch, f.result(None, is_error=True))
    with pytest.raises(LLMTransientError, match="error"):
        await client.generate(system="s", user="u")
    install(monkeypatch, f.result(""))
    with pytest.raises(LLMTransientError, match="empty"):
        await client.generate(system="s", user="u")


def test_tutor_can_use_claude_without_an_api_key(settings: Settings) -> None:
    client = build_llm_client(
        settings.model_copy(update={"llm_provider": "claude", "llm_api_key": None})
    )
    assert isinstance(client, ClaudeAgentLLMClient)
    assert client.model == settings.coach_generation_model
