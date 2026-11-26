"""SCHEMA POST frozen parsers (design v1.0, freeze delta 5).
SELF-TESTING: refuses import if any vector fails. These are the ONLY
parsers used on model output in POST cells.

parse_action: checkup-turn grammar (rotate/move/try_lift/put_down/done)
parse_choice: {"deliver": <name>} against the 3 candidate names
Hardening (schema_l3_score lineage): NOTE lines stripped; the LAST
action-shaped JSON object wins; >1 action-shaped object refused;
word-boundary name matching; ambiguous/non-candidate choices refused.
"""

from __future__ import annotations
import json
import re

ACTION_KEYS = ("rotate", "move", "try_lift", "put_down", "done")


def _objs(reply: str, keys: tuple) -> list:
    body = "\n".join(
        (l for l in reply.splitlines() if not l.strip().upper().startswith("NOTE:"))
    )
    out = []
    for m in re.finditer("\\{[^{}]*\\}", body):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(o, dict) and any((k in o for k in keys)):
            out.append(o)
    return out


def parse_action(reply: str) -> dict:
    objs = _objs(reply, ACTION_KEYS)
    if len(objs) == 0:
        return {"kind": "invalid", "why": "no action object"}
    if len(objs) > 1:
        return {"kind": "invalid", "why": ">1 action object"}
    o = objs[-1]
    if "rotate" in o:
        d = str(o["rotate"]).strip().lower()
        if d not in ("left", "right"):
            return {"kind": "invalid", "why": "bad rotate arg"}
        return {"kind": "rotate", "action": {"rotate": d}}
    if "move" in o:
        return {"kind": "move", "action": {"move": "forward"}}
    if "try_lift" in o:
        return {
            "kind": "try_lift",
            "action": {"try_lift": str(o["try_lift"]).strip().lower()},
        }
    if "put_down" in o:
        return {"kind": "put_down", "action": {"put_down": True}}
    return {"kind": "done"}


def parse_choice(reply: str, names: tuple) -> dict:
    objs = _objs(reply, ("deliver",))
    if objs:
        val = str(objs[-1].get("deliver", "")).strip().lower()
        hits = [n for n in names if re.search(f"\\b{re.escape(n)}\\b", val)]
        if len(hits) == 1:
            return {"choice": hits[0], "how": "json"}
        return {"choice": None, "how": f"json-ambiguous({len(hits)})"}
    body = "\n".join(
        (l for l in reply.splitlines() if not l.strip().upper().startswith("NOTE:"))
    )
    hits = [n for n in names if re.search(f"\\b{re.escape(n)}\\b", body.lower())]
    if len(hits) == 1:
        return {"choice": hits[0], "how": "text-unique"}
    return {"choice": None, "how": f"refused({len(hits)})"}


def _selftest() -> None:
    a = parse_action
    if not a('{"rotate": "left"}')["kind"] == "rotate":
        raise ValueError("Numerical validation failed at source line 76")
    if not a('NOTE: x\n{"move": "forward"}')["kind"] == "move":
        raise ValueError("Numerical validation failed at source line 77")
    if not a('{"try_lift": " Mug "}')["action"]["try_lift"] == "mug":
        raise ValueError("Numerical validation failed at source line 78")
    if not a('{"done": true}')["kind"] == "done":
        raise ValueError("Numerical validation failed at source line 79")
    if not a('{"rotate":"up"}')["kind"] == "invalid":
        raise ValueError("Numerical validation failed at source line 80")
    if not a('{"move":"forward"}\n{"done":true}')["kind"] == "invalid":
        raise ValueError("Numerical validation failed at source line 81")
    if not a("I will look around")["kind"] == "invalid":
        raise ValueError("Numerical validation failed at source line 82")
    if not a('NOTE: {"move":"forward"}')["kind"] == "invalid":
        raise ValueError("Numerical validation failed at source line 83")
    c = parse_choice
    N = ("mug", "pan", "winebottle")
    if not c('{"deliver": "pan"}', N)["choice"] == "pan":
        raise ValueError("Numerical validation failed at source line 86")
    if not c('NOTE: mug\n{"deliver": "the winebottle"}', N)["choice"] == "winebottle":
        raise ValueError("Numerical validation failed at source line 87")
    if not c('{"deliver": "mug or pan"}', N)["choice"] is None:
        raise ValueError("Numerical validation failed at source line 89")
    if not c('{"deliver": "apple"}', N)["choice"] is None:
        raise ValueError("Numerical validation failed at source line 90")
    if not c("I deliver the pan.", N)["choice"] == "pan":
        raise ValueError("Numerical validation failed at source line 91")
    if not c("mug pan winebottle", N)["choice"] is None:
        raise ValueError("Numerical validation failed at source line 92")
    if not c('{"deliver":"pan"}\n{"deliver":"mug"}', N)["choice"] == "mug":
        raise ValueError("Numerical validation failed at source line 93")


_selftest()
