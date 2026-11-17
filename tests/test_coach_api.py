"""The memory coach end to end over HTTP, with the offline coach model."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.assistant.events import TurnRecorder
from app.assistant.runner import AgentOutcome, AgentRequest, AgentRunner
from app.coach.fake import FakeCoachLLM
from app.coach.service import CoachService

NOTES = (
    b"The Preamble of the Constitution was amended only once, by the Forty-second Amendment Act of 1976. "
    b"That amendment inserted the words socialist, secular and integrity into the Preamble. "
    b"The Election Commission of India is a permanent constitutional body established under Article 324. "
    b"The Comptroller and Auditor General audits the accounts of the Union and the States under Article 148. "
    b"Fundamental Duties were added to the Constitution on the recommendation of the Swaran Singh Committee."
)


async def upload(client: AsyncClient, headers: dict[str, str], data: bytes = NOTES) -> str:
    response = await client.post(
        "/documents", files={"file": ("polity.md", data, "text/markdown")}, headers=headers
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def make_deck(client: AsyncClient, headers: dict[str, str], **extra: Any) -> dict[str, Any]:
    body = {"name": "Indian Polity", "track": "upsc"} | extra
    response = await client.post("/decks", json=body, headers=headers)
    assert response.status_code == 201, response.text
    return dict(response.json())


async def generate(
    client: AsyncClient, headers: dict[str, str], coach: CoachService, deck_id: str, **body: Any
) -> dict[str, Any]:
    response = await client.post(f"/decks/{deck_id}/generate", json=body, headers=headers)
    assert response.status_code == 202, response.text
    await coach.wait_for_jobs()
    job = await client.get(f"/jobs/{response.json()['id']}", headers=headers)
    return dict(job.json())


@pytest.fixture
def coach(app: FastAPI) -> CoachService:
    return app.state.coach_service  # type: ignore[no-any-return]


class TestDecks:
    async def test_presets_cover_both_tracks(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        presets = (await client.get("/decks/presets", headers=auth_headers)).json()
        tracks = {p["track"] for p in presets}
        assert {"upsc", "tech"} <= tracks
        assert any("Harbor" in p["description"] for p in presets)

    async def test_crud_and_unique_names(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        deck = await make_deck(client, auth_headers, language="hinglish")
        assert deck["language"] == "hinglish" and deck["desired_retention"] == 0.9
        clash = await client.post("/decks", json={"name": "indian polity"}, headers=auth_headers)
        assert clash.status_code == 400

        patched = await client.patch(
            f"/decks/{deck['id']}", json={"new_per_day": 5}, headers=auth_headers
        )
        assert patched.json()["new_per_day"] == 5
        assert len((await client.get("/decks", headers=auth_headers)).json()) == 1
        assert (
            await client.delete(f"/decks/{deck['id']}", headers=auth_headers)
        ).status_code == 204
        assert (await client.get(f"/decks/{deck['id']}", headers=auth_headers)).status_code == 404

    async def test_decks_are_private(
        self, client: AsyncClient, auth_headers: dict[str, str], other_auth_headers: dict[str, str]
    ) -> None:
        deck = await make_deck(client, auth_headers)
        assert (
            await client.get(f"/decks/{deck['id']}", headers=other_auth_headers)
        ).status_code == 404
        assert (await client.get("/decks", headers=other_auth_headers)).json() == []

    async def test_needs_auth(self, client: AsyncClient) -> None:
        assert (await client.get("/decks")).status_code == 401
        assert (await client.get("/review/queue")).status_code == 401


class TestGeneration:
    async def test_from_a_document(
        self, client: AsyncClient, auth_headers: dict[str, str], coach: CoachService
    ) -> None:
        doc = await upload(client, auth_headers)
        deck = await make_deck(client, auth_headers)
        job = await generate(
            client, auth_headers, coach, deck["id"], document_ids=[doc], max_cards=4
        )
        assert job["status"] == "done", job
        assert job["progress"] == 1.0
        assert job["stats"]["kept"] == 4

        cards = (await client.get(f"/decks/{deck['id']}/cards", headers=auth_headers)).json()
        assert len(cards) == 4
        first = cards[0]
        assert first["source_filename"] == "polity.md"
        assert first["verification"]["verdict"] == "pass"
        assert first["concept"] == "Key facts"
        assert len(first["hints"]) == 3
        assert first["source_excerpt"].split()[0] in NOTES.decode()

    async def test_second_run_does_not_duplicate(
        self, client: AsyncClient, auth_headers: dict[str, str], coach: CoachService
    ) -> None:
        doc = await upload(client, auth_headers)
        deck = await make_deck(client, auth_headers)
        await generate(client, auth_headers, coach, deck["id"], document_ids=[doc], max_cards=5)
        again = await generate(
            client, auth_headers, coach, deck["id"], document_ids=[doc], max_cards=5
        )
        assert again["stats"]["kept"] == 0
        assert again["stats"]["rejected"]["duplicate"] >= 1

    async def test_verifier_rejecting_everything(
        self, client: AsyncClient, auth_headers: dict[str, str], coach: CoachService
    ) -> None:
        coach._models.verifier = FakeCoachLLM(verdict="fail")
        doc = await upload(client, auth_headers)
        deck = await make_deck(client, auth_headers)
        job = await generate(client, auth_headers, coach, deck["id"], document_ids=[doc])
        assert job["status"] == "done"
        assert job["stats"]["kept"] == 0
        assert job["stats"]["rejected"]["failed verification"] >= 1

    async def test_bad_requests(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        other_auth_headers: dict[str, str],
    ) -> None:
        deck = await make_deck(client, auth_headers)
        empty = await client.post(f"/decks/{deck['id']}/generate", json={}, headers=auth_headers)
        assert empty.status_code == 400
        theirs = await upload(client, other_auth_headers)
        stolen = await client.post(
            f"/decks/{deck['id']}/generate", json={"document_ids": [theirs]}, headers=auth_headers
        )
        assert stolen.status_code == 404

    async def test_from_a_topic_via_research(
        self, client: AsyncClient, auth_headers: dict[str, str], coach: CoachService
    ) -> None:
        class Researcher(AgentRunner):
            name = "researcher"

            async def run(
                self, request: AgentRequest, *, search_documents: Any, emitter: Any = None
            ) -> AgentOutcome:
                assert "Write a factual study note on: Bellman equation" in request.question
                recorder = TurnRecorder()
                recorder.final_text = (
                    "The Bellman equation expresses the value of a state as the immediate reward plus the discounted value of the next state. "
                    "The discount factor gamma lies between zero and one and controls how much future rewards matter. "
                    "Q-learning updates action values towards the Bellman optimality target using sampled transitions.\n\nSources: https://example.org"
                )
                await recorder.finish()
                return AgentOutcome(text=recorder.final_text, session_id=None, steps=recorder.steps)

        coach._agent = Researcher()
        deck = await make_deck(client, auth_headers, name="RL", track="tech")
        job = await generate(
            client, auth_headers, coach, deck["id"], topic="Bellman equation", max_cards=3
        )
        assert job["status"] == "done", job
        docs = (await client.get("/documents", headers=auth_headers)).json()
        assert [d["filename"] for d in docs] == ["topic-bellman-equation.md"]
        assert job["stats"]["kept"] == 3

    async def test_failed_research_fails_the_job(
        self, client: AsyncClient, auth_headers: dict[str, str], coach: CoachService
    ) -> None:
        class Broken(AgentRunner):
            async def run(
                self, request: AgentRequest, *, search_documents: Any, emitter: Any = None
            ) -> AgentOutcome:
                return AgentOutcome(
                    text=None, session_id=None, steps=[], is_error=True, error="not signed in"
                )

        coach._agent = Broken()
        deck = await make_deck(client, auth_headers)
        job = await generate(client, auth_headers, coach, deck["id"], topic="Anything")
        assert job["status"] == "failed" and "not signed in" in job["error"]

    async def test_unknown_job(self, client: AsyncClient, auth_headers: dict[str, str]) -> None:
        response = await client.get(
            "/jobs/00000000-0000-0000-0000-000000000000", headers=auth_headers
        )
        assert response.status_code == 404


class TestReview:
    async def _ready(
        self, client: AsyncClient, headers: dict[str, str], coach: CoachService, **deck: Any
    ) -> dict[str, Any]:
        doc = await upload(client, headers)
        made = await make_deck(client, headers, **deck)
        await generate(client, headers, coach, made["id"], document_ids=[doc], max_cards=5)
        return made

    async def test_queue_hides_answers_and_respects_new_limit(
        self, client: AsyncClient, auth_headers: dict[str, str], coach: CoachService
    ) -> None:
        deck = await self._ready(client, auth_headers, coach, new_per_day=3)
        queue = (await client.get("/review/queue", headers=auth_headers)).json()
        assert queue["new_total"] == 3 and queue["due_total"] == 0
        card = queue["cards"][0]
        assert (
            "answer" not in card and card["hint_count"] == 3 and card["deck_name"] == deck["name"]
        )

    async def test_full_review_cycle(
        self, client: AsyncClient, auth_headers: dict[str, str], coach: CoachService
    ) -> None:
        await self._ready(client, auth_headers, coach)
        queue = (await client.get("/review/queue", headers=auth_headers)).json()["cards"]
        first, second, third = queue[0]["id"], queue[1]["id"], queue[2]["id"]
        answers = {
            c["id"]: c
            for c in (
                await client.get(f"/decks/{queue[0]['deck_id']}/cards", headers=auth_headers)
            ).json()
        }

        hint = (await client.get(f"/cards/{first}/hint?level=2", headers=auth_headers)).json()
        assert hint["level"] == 2 and hint["remaining"] == 1

        right = await client.post(
            f"/cards/{first}/review",
            json={"response": answers[first]["answer"], "duration_ms": 30_000},
            headers=auth_headers,
        )
        body = right.json()
        assert right.status_code == 200 and body["correct"] and body["rating"] == "good"
        assert body["source_filename"] == "polity.md" and body["explanation"]

        wrong = (
            await client.post(
                f"/cards/{second}/review", json={"response": "zzz"}, headers=auth_headers
            )
        ).json()
        assert not wrong["correct"] and wrong["rating"] == "again" and wrong["interval_days"] < 1

        revealed = (
            await client.post(
                f"/cards/{third}/review", json={"revealed": True}, headers=auth_headers
            )
        ).json()
        assert revealed["grader"] == "revealed" and revealed["rating"] == "again"

        stats = (await client.get("/stats", headers=auth_headers)).json()
        assert stats["reviewed_today"] == 3
        assert stats["streak_days"] == 1
        deck_stats = stats["decks"][0]
        assert deck_stats["new"] == 2 and deck_stats["learning"] == 3
        assert deck_stats["concepts"][0]["studied"] == 3

        # Failed cards come back after their 1-minute learning step; the
        # correct one waits for its 10-minute step.
        user_id = await _user_id(coach, queue[0]["deck_id"])
        later = datetime.now(tz=UTC) + timedelta(minutes=5)
        due = [i for i in await coach.queue(user_id=user_id, limit=50, now=later) if not i.is_new]
        assert {str(i.card.id) for i in due} == {second, third}

    async def test_cards_are_private_and_editable(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        other_auth_headers: dict[str, str],
        coach: CoachService,
    ) -> None:
        deck = await self._ready(client, auth_headers, coach)
        card = (await client.get(f"/decks/{deck['id']}/cards", headers=auth_headers)).json()[0]
        assert (
            await client.get(f"/cards/{card['id']}", headers=other_auth_headers)
        ).status_code == 404
        assert (
            await client.post(
                f"/cards/{card['id']}/review", json={"response": "x"}, headers=other_auth_headers
            )
        ).status_code == 404

        edited = await client.patch(
            f"/cards/{card['id']}",
            json={"explanation": "Better explanation."},
            headers=auth_headers,
        )
        assert edited.json()["explanation"] == "Better explanation."

        flagged = await client.post(
            f"/cards/{card['id']}/flag", json={"reason": "Wrong year"}, headers=auth_headers
        )
        assert (
            flagged.json()["status"] == "flagged" and flagged.json()["flag_reason"] == "Wrong year"
        )
        queue = (await client.get("/review/queue?limit=50", headers=auth_headers)).json()["cards"]
        assert card["id"] not in {c["id"] for c in queue}
        blocked = await client.post(
            f"/cards/{card['id']}/review", json={"response": "x"}, headers=auth_headers
        )
        assert blocked.status_code == 400

        restored = await client.patch(
            f"/cards/{card['id']}", json={"status": "active"}, headers=auth_headers
        )
        assert restored.json()["status"] == "active"
        assert (
            await client.delete(f"/cards/{card['id']}", headers=auth_headers)
        ).status_code == 204
        assert (await client.get(f"/cards/{card['id']}", headers=auth_headers)).status_code == 404

    async def test_mcq_needs_a_choice_and_answer_index_is_validated(
        self, client: AsyncClient, auth_headers: dict[str, str], coach: CoachService
    ) -> None:
        deck = await self._ready(client, auth_headers, coach)
        card = (await client.get(f"/decks/{deck['id']}/cards", headers=auth_headers)).json()[0]
        made_mcq = await client.patch(
            f"/cards/{card['id']}",
            json={"choices": ["14", "19", "21", "32"], "answer_index": 2, "answer": "x"},
            headers=auth_headers,
        )
        assert made_mcq.status_code == 200
        # Kind stays cloze here; change kind via the service to exercise MCQ review.
        await coach.update_card(
            uuid.UUID(made_mcq.json()["id"]), await _user_id(coach, deck["id"]), kind="mcq"
        )
        missing = await client.post(f"/cards/{card['id']}/review", json={}, headers=auth_headers)
        assert missing.status_code == 400
        ok = await client.post(
            f"/cards/{card['id']}/review",
            json={"choice_index": 2, "duration_ms": 2000},
            headers=auth_headers,
        )
        assert (
            ok.json()["correct"] and ok.json()["rating"] == "easy" and ok.json()["answer"] == "21"
        )
        bad = await client.patch(
            f"/cards/{card['id']}", json={"answer_index": 9}, headers=auth_headers
        )
        assert bad.status_code == 400

    async def test_deleting_a_deck_removes_everything(
        self, client: AsyncClient, auth_headers: dict[str, str], coach: CoachService
    ) -> None:
        deck = await self._ready(client, auth_headers, coach)
        card = (await client.get(f"/decks/{deck['id']}/cards", headers=auth_headers)).json()[0]
        await client.post(
            f"/cards/{card['id']}/review", json={"response": "x"}, headers=auth_headers
        )
        assert (
            await client.delete(f"/decks/{deck['id']}", headers=auth_headers)
        ).status_code == 204
        assert (await client.get(f"/cards/{card['id']}", headers=auth_headers)).status_code == 404
        assert (await client.get("/review/queue", headers=auth_headers)).json()["cards"] == []


async def _user_id(coach: CoachService, deck_id: str) -> Any:
    from app.db.models import Deck

    async with coach._database.session() as session:
        deck = await session.get(Deck, uuid.UUID(deck_id))
        assert deck is not None
        return deck.user_id
