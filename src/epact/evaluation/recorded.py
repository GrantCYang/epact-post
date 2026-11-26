"""EPACT scoring semantics for compact archived episode records."""

from collections import Counter
from research_common.statistics import paired


def opened(row):
    target = row["receptacle_id"]
    return any(
        (
            s.get("action") == "Open"
            and (s.get("args") or {}).get("target") == target
            and (s.get("ok") is True)
            and any(
                (
                    n.get("action") == "OpenObject"
                    and n.get("success") is True
                    and (n.get("objectId") == target)
                    for n in s.get("native_steps") or []
                )
            )
            for s in row["open_steps"]
        )
    )


def episode_summary(rows):
    for r in rows:
        if r["scored_correct"] != (bool(r["correct"]) and r["forced_answer"] is None):
            raise ValueError("Archived scoring semantics do not match")
    scores = paired(rows, "scored_correct")
    return {
        "pairs": len(scores),
        "pair_success": sum(scores.values()),
        "episodes": len(rows),
        "scored_correct": sum((r["scored_correct"] for r in rows)),
        "queried_container_opened_eps": sum((opened(r) for r in rows)),
        "correct_after_opening": sum((opened(r) and r["scored_correct"] for r in rows)),
        "invalid_terminations": sum((r["status"] == "invalid_action" for r in rows)),
        "answer_counts": dict(
            sorted(Counter((r["answer"] or "(missing)" for r in rows)).items())
        ),
    }


def witness_scores(rows, pred="pred"):
    values = [
        {**r, "gold": r["answer"], "correct": r[pred] == r["answer"]} for r in rows
    ]
    pairs = paired(values)
    return (pairs, {r["task_id"]: r["correct"] for r in values})
