"""Eval harness for the LangGraph extraction agents — run one against a real transcript.

Built for the data_model_extractor, but works for any extractor in the pipeline.

Usage:
    # free: print the exact prompt and segment plan, make no API call
    venv/Scripts/python scripts/eval_extractors.py --dry-run

    # run the built-in sample meeting, segmented like realtime transcription
    venv/Scripts/python scripts/eval_extractors.py

    # run your own transcript, whole-transcript instead of segmented
    venv/Scripts/python scripts/eval_extractors.py --transcript path/to/meeting.txt --whole

    # a different extractor
    venv/Scripts/python scripts/eval_extractors.py --extractor features

Segmented mode is the default because it is what production actually does: the
worker calls the pipeline per finalized transcript segment, never on the whole
meeting. Entities therefore arrive partial and repeated, and _merge_entities has
to reassemble them — this harness shows both the raw per-segment output and the
merged result so you can see whether the merge is doing its job.

Prints token counts, not costs — check current pricing yourself rather than
trusting a number hardcoded here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Transcripts and separators are full of em dashes and box-drawing characters;
# the default Windows console codepage (cp1252) cannot encode either.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.ai.pipeline import EXTRACTORS  # noqa: E402
from app.config import get_settings  # noqa: E402

# A planning meeting with: clear entities, one vague stretch that implies nothing
# storable, and an off-topic aside. The last two exist to test the prompt's
# "do not invent a schema" rule — a good run extracts nothing from them.
_SAMPLE = """
Okay so the core of it is invoices. Each invoice belongs to one customer, has an
amount, and a due date. Status is either draft, sent, or paid — those three only.

Right, and a customer has a name, an email, and a billing address. One customer
can obviously have many invoices over time.

Do we need line items? Yeah, each invoice breaks down into line items — a
description, a quantity, and a unit price. That's how we get the total.

We should think about the general direction here, whether this is the right shape
for the product long term. I think there's a bigger conversation to have about
positioning but let's not get into it now.

Oh and the deploy is still broken from Friday, I'll look at it after this.

One more thing — users log in with email, and each user only sees their own
invoices. That's important, we can't have people seeing each other's billing.
""".strip()


def _segment(text: str, words_per_segment: int) -> list[str]:
    """Chop a transcript into realtime-sized chunks, mirroring the worker's input."""
    words = text.split()
    return [
        " ".join(words[i : i + words_per_segment])
        for i in range(0, len(words), words_per_segment)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, help="transcript file (default: built-in sample)")
    parser.add_argument(
        "--extractor", default="entities", choices=sorted(EXTRACTORS),
        help="which extractor to run (default: entities)",
    )
    parser.add_argument("--whole", action="store_true", help="one pass over the full transcript")
    parser.add_argument(
        "--words", type=int, default=60,
        help="words per segment in segmented mode (default: 60)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the prompt and segment plan, make no API call",
    )
    args = parser.parse_args()

    transcript = args.transcript.read_text(encoding="utf-8").strip() if args.transcript else _SAMPLE
    segments = [transcript] if args.whole else _segment(transcript, args.words)

    module_name = EXTRACTORS[args.extractor]
    agent = __import__(f"app.ai.agents.{module_name}", fromlist=["run"])

    settings = get_settings()
    model = settings.ANTHROPIC_MODEL

    print(f"extractor : {args.extractor} ({module_name})")
    print(f"model     : {model}")
    print(f"transcript: {len(transcript)} chars, {len(transcript.split())} words")
    print(f"mode      : {'whole transcript' if args.whole else f'segmented, {args.words} words'}")
    print(f"segments  : {len(segments)}\n")

    if args.dry_run:
        print("─" * 72)
        print("SYSTEM PROMPT")
        print("─" * 72)
        print(agent._SYSTEM)
        print("\n" + "─" * 72)
        print("SEGMENTS THAT WOULD BE SENT (one API call each)")
        print("─" * 72)
        for i, seg in enumerate(segments, 1):
            print(f"\n[{i}/{len(segments)}] {seg}")
        print(f"\n{len(segments)} API call(s) would be made. Re-run without --dry-run to spend them.")
        return 0

    if not settings.ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY is not set — nothing to run.", file=sys.stderr)
        return 1

    usage: list[dict] = []
    raw_results: list[dict] = []
    for i, seg in enumerate(segments, 1):
        print("─" * 72)
        print(f"[{i}/{len(segments)}] {seg[:100]}{'…' if len(seg) > 100 else ''}")
        try:
            result = agent.run(seg, settings.ANTHROPIC_API_KEY, model, usage=usage)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        raw_results.append(result)
        print(json.dumps(result, indent=2))

    print("\n" + "═" * 72)
    print("TOKEN USAGE")
    print("═" * 72)
    print(f"input : {sum(u['input_tokens'] for u in usage)}")
    print(f"output: {sum(u['output_tokens'] for u in usage)}")
    print(f"calls : {len(usage)}")

    # The merge only applies to entities — the other extractors emit flat lists
    # that the aggregator concatenates as-is.
    if args.extractor == "entities":
        from app.build.blueprint_generator import _merge_entities

        merged = _merge_entities(
            entity for result in raw_results for entity in result.get("entities", [])
        )
        raw_count = sum(len(r.get("entities", [])) for r in raw_results)
        print("\n" + "═" * 72)
        print(f"MERGED — {raw_count} raw entities across segments → {len(merged)} distinct")
        print("═" * 72)
        print(json.dumps(merged, indent=2))
        print("\nCheck: are the names stable across segments (Invoice vs Invoices vs invoice)?")
        print("Did the vague and off-topic segments correctly yield nothing?")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
