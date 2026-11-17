# Lelouch: the memory coach

Lelouch turns what you study into verified questions and brings you back to
recall them at the right time, until you can answer without help.

## Why this product

- Plain chatbots act as a crutch. In a field experiment with about 1,000 high
  school students, grades rose while students had GPT-4 and fell 17% below
  the control group once access was removed (Bastani et al., *PNAS* 2025).
- A tutor built on learning science roughly doubled learning gains over an
  excellent active-learning class (Kestin et al., *Scientific Reports* 2025).
- The two highest-utility study techniques are practice testing and spaced
  (distributed) practice (Dunlosky et al., *Psychological Science in the
  Public Interest* 2013; Roediger & Karpicke 2006).
- ChatGPT Study Mode, Gemini Guided Learning and Claude Learning Mode all
  withhold answers, but none schedules spaced review or reaches out to the
  learner. That is the gap Lelouch fills.

## Decisions

| Question | Decision |
|---|---|
| First audiences | **UPSC (Prelims and Mains)** and **AI engineering**: AI, ML, DL, RL, RL environments and evaluation frameworks such as Harbor |
| Push channel | **Telegram** first; WhatsApp in the public phase |
| Language | **English** by default; **Hindi** and **Hinglish** per deck |
| Old modes | Lelouch becomes the coach. Agent chat stays as "Ask Lelouch". The tutor's solve-and-check engine becomes the hint and verification engine behind the cards |

## Done means

1. Upload a chapter or name a topic and get verified questions in about a minute.
2. Review in the web app: attempt, hint, reveal, graded result.
3. The next day, a Telegram message brings a short review; answer in the chat.
4. See strong and weak concepts and a weekly "without hints" score.

## Architecture

```
Your material -> concept extraction -> question writing -> verifier -> cards
(docs, topics)        (Claude)            (Claude)      (vs source)    |
                                                                       v
Telegram / web <- daily scheduler <- FSRS memory model <- review log <- grader
   (push)          (who is due)        (per card)        (every answer)
```

## Milestones

### M2: Learning engine (backend) - built
- Tables: decks, concepts, cards, card_states (FSRS memory), review_logs,
  generation_jobs.
- Question types, cheapest to grade first: multiple choice (including UPSC
  statement-style questions), fill-in-the-blank, numeric, short answer.
- Generation pipeline: passages -> concepts -> 2-4 questions each -> checks
  (format, supporting quote really in the source, duplicates) -> Claude
  verifier -> stored with the passage they came from.
- Topic decks: the agent researches the topic, writes a sourced study note
  into your documents, and cards are made from that note.
- Grading: exact for choices, blanks and numbers; Claude for short answers.
  Answers map to FSRS ratings (Again, Hard, Good, Easy); target 90% recall.
- Hint ladder: three levels, written at generation time.
- One Claude client (Agent SDK, no tools) for generation, verification and
  grading; works with the Max login now and an API key later.
- API: decks, presets, generate, jobs, cards, review queue, hints, review,
  stats. CLI: `scripts/generate_cards.py`.

### M3: Review experience (web app)
Today screen, review session, library with concept strength, create flow,
progress view. Agent chat becomes "Ask Lelouch", opened from any card.
Tutor mode is removed from the UI.

### M4: Push (Telegram)
Morning job in the user's timezone with quiet hours. Telegram bot linked by a
one-time code; choices as buttons, typed answers, hint button.

### M5: Proof and motivation
Weekly unassisted check, exam date planning (UPSC Prelims and Mains dates),
streaks, Hindi and Hinglish polish, Mains answer-writing practice.

### M6: Ready for other people
`ANTHROPIC_API_KEY` (required before anyone else uses it), model routing and
per-user cost caps, real sign-in, hosting with managed PostgreSQL + pgvector,
backups, error tracking, data export and delete, legal advice on consent for
users under 18.

### M7: Pilot
20-50 learners for four weeks. Measure active days, reviews, recall accuracy,
flagged-question rate and change in the unassisted score.
