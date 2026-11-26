"""Frozen action-only prompt and parser used by benchmark and diagnostics."""

import json
import re

PROMPT = 'You are an embodied agent in a house, answering: {query}\nThe image is your current view. Choose ONE next action as JSON.\nActions: GoTo(target: room or object id), Open(target), Close(target), Toggle(target), MoveAside(target), LookAt(target), Answer(answer: "yes"/"no", confidence: 0..1).\nObject ids visible earlier appear in the history. Remaining budget: {budget}. Every action costs budget; answer as soon as you can determine it.\nHistory of your previous actions and results:\n{history}\nReply with JSON only, e.g. {{"action": "Open", "args": {{"target": "Fridge|2|1"}}}}'
MAX_STEPS = 14


def parse_action(text: str) -> tuple[str, dict] | None:
    match = re.search("\\{.*\\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    name = obj.get("action")
    args = obj.get("args") or {}
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return (name, args)


def policy_prompt(state, history, final=False, hide_budget=False):
    hist_text = (
        "\n".join(
            (
                f"{i}. {h['action']}({h['args']}) → {('ok' if h['ok'] else 'FAILED: ' + h.get('error', '')[:60])}"
                for i, h in enumerate(history[-6:])
            )
        )
        or "(none yet)"
    )
    text = PROMPT.format(
        query=state.get("query", ""),
        budget="not disclosed"
        if hide_budget
        else state.get("remaining_budget", state.get("budget")),
        history=hist_text,
    )
    if state.get("protocol") in ("inspection-v1", "inspection-v4-dev"):
        text = text.replace(
            "GoTo(target: room or object id)",
            "GoTo(target, view: near/inspect/wide, side: auto/north/east/south/west)",
        )
        text = text.replace(
            "LookAt(target)", "LookAt(target, region: center/upper/lower)"
        )
        text = text.replace(
            "GoTo(target, view: near/inspect/wide, side: auto/north/east/south/west)",
            "GoTo(target, view: near/inspect/wide, side: auto/north/east/south/west, stance: stand/crouch)",
        )
        text = text.replace(
            '{"action": "Open", "args": {"target": "Fridge|2|1"}}',
            '{"action": "GoTo", "args": {"target": "query_location"}}',
        )
        text = text.replace(
            "Object ids visible earlier appear in the history.",
            "Answer using visual evidence; inspect further if the current view is insufficient.",
        )
        text += "\nReference and action rules:\n" + state["reference_help"]
    if final:
        text += "\nYou can now determine the answer. Reply with the Answer action."
    return text


REASON_SWAP = "First think briefly - two to four short sentences about what you already know and what evidence you still need. Then end your reply with ONE action JSON on its own line"
OBJ_RE = re.compile("\\{(?:[^{}]|\\{[^{}]*\\})*\\}", re.DOTALL)


def parse_action_reason(text: str):
    for m in reversed(list(OBJ_RE.finditer(text))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("action"), str)
            and isinstance(obj.get("args") or {}, dict)
        ):
            return (obj["action"], obj.get("args") or {})
    return None
