"""Claude as a plain text-in, text-out model, through the Claude Agent SDK.

Used for structured jobs - writing, verifying and grading cards, and the
tutor graph's nodes - where an agent loop would only add cost. The engine runs
with **no tools** and a single turn.

It authenticates like the agent: your own ``claude`` login (billed to your
plan's Agent SDK credit, personal use only) or ``ANTHROPIC_API_KEY`` when set.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.llm.base import LLMClient, LLMPermanentError, LLMResponse, LLMTransientError
from app.logging_config import get_logger

logger = get_logger(__name__)


class ClaudeAgentLLMClient(LLMClient):
    name = "claude"
    supports_streaming = False

    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: float = 240.0,
        workspace: str = "./workspace/_llm",
        cli_path: str | None = None,
    ) -> None:
        self.model = model
        self._timeout = timeout_seconds
        self._workspace = Path(workspace).resolve()
        self._cli_path = cli_path
        #: Running total for this client, in USD at API list prices.
        self.cost_usd = 0.0

    def _options(self, system: str) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        self._workspace.mkdir(parents=True, exist_ok=True)
        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=system,
            tools=[],
            allowed_tools=[],
            permission_mode="dontAsk",
            max_turns=1,
            setting_sources=[],
            strict_mcp_config=True,
            mcp_servers={},
            thinking={"type": "disabled"},  # type: ignore[arg-type]
            cwd=self._workspace,
            cli_path=self._cli_path,
        )

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> LLMResponse:
        # temperature / max_tokens are not exposed by the engine; the prompts
        # ask for compact JSON instead.
        del temperature, max_tokens
        from claude_agent_sdk import AssistantMessage, CLINotFoundError, ResultMessage, TextBlock
        from claude_agent_sdk import query as sdk_query

        texts: list[str] = []
        result: ResultMessage | None = None
        try:
            async with asyncio.timeout(self._timeout):
                async for message in sdk_query(prompt=user, options=self._options(system)):
                    if isinstance(message, AssistantMessage):
                        texts.extend(b.text for b in message.content if isinstance(b, TextBlock))
                    elif isinstance(message, ResultMessage):
                        result = message
        except TimeoutError as exc:
            raise LLMTransientError(f"Claude did not answer within {self._timeout:g}s") from exc
        except CLINotFoundError as exc:
            raise LLMPermanentError("The Claude engine was not found") from exc
        except Exception as exc:  # noqa: BLE001 - engine/process errors
            lowered = str(exc).lower()
            if "login" in lowered or "auth" in lowered or "401" in lowered:
                raise LLMPermanentError(
                    "The Claude engine is not signed in. Run `claude` and /login, "
                    "or set ANTHROPIC_API_KEY."
                ) from exc
            raise LLMTransientError(f"Claude call failed: {exc}") from exc

        if result is not None and result.total_cost_usd:
            self.cost_usd += result.total_cost_usd
        if result is not None and result.is_error:
            raise LLMTransientError(f"Claude returned an error: {result.subtype}")
        text = (result.result if result is not None and result.result else "\n".join(texts)).strip()
        if not text:
            raise LLMTransientError("Claude returned an empty reply")
        return LLMResponse(
            text=text,
            model=self.model,
            raw={"cost_usd": result.total_cost_usd if result else None},
        )
