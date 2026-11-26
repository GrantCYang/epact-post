"""Probe-T scene serialization (plan §5): fixed format, knowledge not truth."""

from __future__ import annotations


def scene_text(task: dict) -> str:
    summary = task["scene_summary"]
    lines = [f"The house has {len(summary['rooms'])} rooms."]
    for room in summary["rooms"]:
        light = "lit" if room.get("lit", True) else "DARK (you cannot see inside)"
        lines.append(f"Room {room['id']} ({room['type']}), {light}:")
        if not room["receptacles"]:
            lines.append("  (no storage or surfaces)")
        for rec in room["receptacles"]:
            bits = [f"  {rec['id']} ({rec['type']}"]
            if rec["openable"]:
                bits.append(", open" if rec["is_open"] else ", closed")
            bits.append(")")
            entry = "".join(bits)
            if not rec["visible"]:
                entry += ": not seen yet"
            elif rec["openable"] and (not rec["is_open"]):
                entry += ": contents unknown"
            elif rec["visible_contents"]:
                entry += ": visible items: " + ", ".join(rec["visible_contents"])
            else:
                entry += ": nothing visible on/in it"
            lines.append(entry)
    lines.append(f"You are standing in room {task['start_room']}.")
    return "\n".join(lines)


def revealing_first_action(task: dict) -> dict | None:
    """The oracle's first *revealing* action (approach GoTo is not revealing
    unless distance is the hiding kind)."""
    plan = task["oracle_plan"]
    if not plan:
        return None
    kinds = task["hiding_kinds"]
    if "D" in kinds or (kinds == ["L"] and plan[0]["action"] == "GoTo"):
        return plan[0]
    for step in plan:
        if step["action"] in ("Open", "Toggle", "MoveAside"):
            return step
    return plan[0]
