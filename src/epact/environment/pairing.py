"""Branch-independent twin construction; validity is checked after replay.

Pure task/metadata logic. Geometric visibility is not an exclusion criterion:
L can be geometrically visible in a dark frame, and N is intentionally visible.
"""

from copy import deepcopy

STATE_ACTIONS = {
    ("isOpen", True): "CloseObject",
    ("isOpen", False): "OpenObject",
    ("isToggled", True): "ToggleObjectOff",
    ("isToggled", False): "ToggleObjectOn",
}
EMISSIVE = {
    "FloorLamp",
    "DeskLamp",
    "TableLamp",
    "LightSwitch",
    "Candle",
    "Television",
    "Laptop",
    "CellPhone",
}


def family(task):
    kinds = task["hiding_kinds"]
    if task["query_type"] == "STATE":
        return "STATE|" + "+".join(kinds)
    return "+".join(kinds)


def construct_pair(task, objects):
    """Return (two fresh tasks, diagnostic), without changing input objects.

    objects must come from THIS task's setup, not the final scene of some other
    generator branch. Existence is checked against every same-type instance
    within the specified receptacle. Render matching is a later check.
    """
    base = deepcopy(task)
    if "O" in base.get("hiding_kinds", []):
        return ([], {"status": "excluded_scope", "reason": "O explicitly excluded"})
    if base.get("trap") and base.get("hiding_kinds") != ["N"]:
        return (
            [],
            {"status": "excluded_scope", "reason": "non-N trap is a separate control"},
        )
    if base["task_id"].endswith("-twin"):
        return ([], {"status": "duplicate", "reason": "source twin"})
    by_id = {o["objectId"]: o for o in objects}
    twin = deepcopy(base)
    oid = base.get("target_object_id")
    if base["query_type"] == "STATE":
        obj = by_id.get(oid)
        field = base.get("state_field")
        if not obj or field not in obj:
            return (
                [],
                {"status": "unresolved", "reason": "missing state object/property"},
            )
        if obj["objectType"] in EMISSIVE:
            return (
                [],
                {
                    "status": "unresolved",
                    "reason": "emissive STATE needs a separate spill-light design",
                },
            )
        actual = bool(obj[field])
        if actual != base.get("state_value"):
            return (
                [],
                {"status": "invalid", "reason": "source state differs from live setup"},
            )
        action = STATE_ACTIONS.get((field, actual))
        if not action:
            return (
                [],
                {"status": "unresolved", "reason": "unsupported state intervention"},
            )
        twin["state_value"] = not actual
        mod = twin.setdefault("house_mod", {}) or {}
        twin["house_mod"] = mod
        mod["prep"] = list(mod.get("prep") or []) + [
            dict(action=action, objectId=oid, forceAction=True)
        ]
    else:
        if base["answer"] != "yes":
            return (
                [],
                {
                    "status": "not_seed",
                    "reason": "presence twins originate from positive tasks",
                },
            )
        obj = by_id.get(oid)
        rid = base.get("receptacle_id")
        rec = by_id.get(rid)
        if not obj or not rec:
            return (
                [],
                {
                    "status": "invalid",
                    "reason": "positive object or queried location missing",
                },
            )
        target_type = base.get("target_type")
        if obj["objectType"] != target_type or rid not in (
            obj.get("parentReceptacles") or []
        ):
            return (
                [],
                {
                    "status": "invalid",
                    "reason": "positive label or location relation incorrect",
                },
            )
        drop = sorted(
            (
                o["objectId"]
                for o in objects
                if o["objectType"] == target_type
                and rid in (o.get("parentReceptacles") or [])
            )
        )
        mod = twin.setdefault("house_mod", {}) or {}
        twin["house_mod"] = mod
        mod["removed_objects"] = sorted(
            set(mod.get("removed_objects") or []) | set(drop)
        )
        twin["target_object_id"] = None
        twin["reveal_condition"] = (
            "already_visible"
            if base["hiding_kinds"] == ["N"]
            else "lit_and_visible"
            if "L" in base["hiding_kinds"]
            else "receptacle_open_visible"
            if rec.get("openable")
            else "receptacle_visible"
        )
        twin["oracle_plan"] = [
            a for a in twin.get("oracle_plan", []) if a.get("target") not in drop
        ]
    pid = base["task_id"]
    twin["task_id"] = pid + "-twin"
    twin["answer"] = "no" if base["answer"] == "yes" else "yes"
    for member in (base, twin):
        member["pair_id"] = pid
        member["benchmark_family"] = family(base)
        member["benchmark_condition"] = (
            "already_answerable" if base["hiding_kinds"] == ["N"] else "hidden"
        )
        member["pairing_version"] = "branch-complete-v4"
        member["source_task_id"] = pid
        member["semantic_status"] = "review_pending"
    twin["checks"] = {**twin.get("checks", {}), "paired_twin_of": pid}
    return (
        [base, twin],
        {"status": "constructed", "reason": "requires paired live validation"},
    )
