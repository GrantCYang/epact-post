"""Executed action-chain diagnostics shared by evaluation and training."""

from collections import Counter
import json


def open_stats(row: dict, task: dict) -> dict:
    """Container-open chain fields (episode_stats logic, no_open
    experiment lineage)."""
    steps = row.get("step_log") or []
    attempted = [d for d in row["decided"] if d["action"] == "Open"]
    container = [
        s
        for s in steps
        if s.get("action") == "Open"
        and (s.get("args") or {}).get("target") == task["receptacle_id"]
    ]
    successful = [
        s
        for s in container
        if s.get("ok") is True
        and any(
            (
                n.get("action") == "OpenObject"
                and n.get("success") is True
                and (n.get("objectId") == task["receptacle_id"])
                for n in s.get("native_steps", [])
            )
        )
    ]
    return {
        "open_attempted": bool(attempted),
        "queried_container_opened": bool(successful),
    }


def behavior(decided):
    actions = [d for d in decided if d["action"] != "Answer"]
    keys = [json.dumps([d["action"], d["args"]], sort_keys=True) for d in actions]
    return {
        "no_action": not actions,
        "repeated_actions": len(keys) - len(set(keys)),
        "action_counts": dict(Counter((d["action"] for d in decided))),
    }
