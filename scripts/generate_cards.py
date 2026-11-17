"""Make verified cards from one file, from the command line.

A quick way to judge card quality before the review UI exists:

    python -m scripts.generate_cards notes/polity-ch3.pdf --deck "Indian Polity" --track upsc
    python -m scripts.generate_cards rl-notes.md --deck "RL" --track tech --cards 10
    python -m scripts.generate_cards --topic "Fundamental Duties" --deck "Polity" --track upsc

It uses the same database and settings as the API (your .env), signs in as
--user, uploads the file to that user's documents, runs generation and prints
every card with its answer, hints, verifier verdict and the job's rejection
counts. Real runs call Claude: about 2 model calls per passage.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.assistant.runner import build_agent_runner
from app.coach.models import build_coach_models
from app.coach.service import CoachService, Invalid
from app.config import get_settings
from app.db.repository import UserRepository
from app.db.session import Database
from app.knowledge.embeddings import build_embedder
from app.knowledge.service import KnowledgeService
from app.models.enums import DeckLanguage, Track


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    database = Database(settings.database_url)
    knowledge = KnowledgeService(
        settings=settings, database=database, embedder=build_embedder(settings)
    )
    coach = CoachService(
        settings=settings,
        database=database,
        knowledge=knowledge,
        models=build_coach_models(settings),
        agent_runner=build_agent_runner(settings) if args.topic else None,
    )
    try:
        async with database.session() as session:
            user = await UserRepository(session).get_or_create(args.user)
        decks = {d.name.lower(): d for d in await coach.list_decks(user.id)}
        deck = decks.get(args.deck.lower())
        if deck is None:
            deck = await coach.create_deck(
                user_id=user.id,
                name=args.deck,
                track=Track(args.track),
                language=DeckLanguage(args.language),
            )
            print(f"Created deck {deck.name!r} ({deck.track}, {deck.language})")

        document_ids = []
        if args.file:
            path = Path(args.file)
            document = await knowledge.ingest(
                user_id=user.id,
                filename=path.name,
                content_type=None,
                data=await asyncio.to_thread(path.read_bytes),
            )
            print(f"Added {path.name}: {document.chunk_count} passages indexed")
            document_ids.append(document.id)

        job = await coach.start_generation(
            deck_id=deck.id,
            user_id=user.id,
            document_ids=document_ids or None,
            topic=args.topic,
            focus=args.focus,
            max_cards=args.cards,
        )
        print("Writing and checking questions", end="", flush=True)
        while True:
            await asyncio.sleep(2)
            job = await coach.get_job(job.id, user.id)
            print(".", end="", flush=True)
            if job.status in ("done", "failed"):
                break
        print()
        if job.status == "failed":
            print(f"\nFailed: {job.error}")
            return 1

        cards = [c for c in await coach.list_cards(deck.id, user.id) if c.job_id == job.id]
        for n, card in enumerate(cards, start=1):
            print(f"\n{n}. [{card.kind}] {card.prompt}")
            for i, choice in enumerate(card.choices or []):
                mark = "*" if i == card.answer_index else " "
                print(f"     {mark} {chr(97 + i)}) {choice}")
            print(f"   Answer: {card.answer}")
            for level, hint in enumerate(card.hints, start=1):
                print(f"   Hint {level}: {hint}")
            if card.explanation:
                print(f"   Why: {card.explanation}")
            verdict = card.verification or {}
            checked = f"{verdict.get('verdict')} ({verdict.get('confidence')})"
            print(f"   Checked: {checked}, from {card.source_filename}")
        stats = job.stats
        kept, drafted, passages = stats.get("kept"), stats.get("drafted"), stats.get("passages")
        print(f"\nKept {kept} of {drafted} drafts from {passages} passages.")
        if stats.get("rejected"):
            print("Dropped: " + ", ".join(f"{k} {v}" for k, v in stats["rejected"].items()))
        if stats.get("cost_usd"):
            print(f"Cost: about ${stats['cost_usd']:.2f} of credit")
        return 0
    except Invalid as exc:
        print(f"Error: {exc}")
        return 2
    finally:
        await database.dispose()


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", nargs="?", help="PDF, Word, Markdown or text file")
    parser.add_argument("--topic", help="Research a topic instead of (or as well as) a file")
    parser.add_argument("--deck", required=True)
    parser.add_argument("--track", choices=[t.value for t in Track], default="general")
    parser.add_argument("--language", choices=[lang.value for lang in DeckLanguage], default="en")
    parser.add_argument("--focus", help="What to emphasise")
    parser.add_argument("--cards", type=int, default=12)
    parser.add_argument("--user", default="kanishk", help="Whose deck this is (the sign-in name)")
    args = parser.parse_args(argv)
    if not args.file and not args.topic:
        parser.error("give a file or --topic")
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse(sys.argv[1:]))))
