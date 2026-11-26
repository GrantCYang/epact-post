"""POST scoring semantics for compact archived episode records."""

from post.report import evidence
from post.score import parse_choice
from research_common.statistics import paired


def post_summary(rows):
    computed = []
    for r in rows:
        choice = r["choice"]
        ev = evidence(r["actions"] or [], r["candidates"])
        computed.append(
            {
                **r,
                "correct": choice == r["label_choice"],
                "entailed": bool(ev["entailed"]),
            }
        )
    scores = paired(computed, member="member")
    return {
        "pairs": len(scores),
        "pair_correct": sum(scores.values()),
        "episodes": len(rows),
        "correct_choices": sum((r["correct"] for r in computed)),
        "sufficient_evidence_episodes": sum((r["entailed"] for r in computed)),
        "invalid_turns": sum((r["turn_kinds"].count("invalid") for r in rows)),
        "executed_actions": sum((len(r["actions"] or []) for r in rows)),
    }
