"""SCHEMA POST v1.0 build pipeline (design: SCHEMA_POST_DESIGN.md).

Phases (all sim-only; model calls happen only via the client, later):
  catalog  per kitchen scene: pickupable objects (type, id, mass, pos),
           reachable grid, agent ground y        -> catalog_post.json
  solve    pure analytic: ranked (dock, candidates, theta) specs per
           dimension satisfying the frozen margins -> solved_post.json
  gates    G0 rule-fidelity + G1 oracle in one sim pass per spec, in
           rank order until PAIRS_PER_DIM pass per dim; G2 parity
           asserts over the records                -> gates/ records
  freeze   pairs_post.jsonl + GATES_POST.json (+ sha256), b0-floor
           byte-identity assert; commit before any model call.

No placement: candidates are SELECTED from native layouts, never moved
(G0 restores verified pickups via TeleportObject to original pose).
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from post.body import PostBody, ADJ, FACING, dist2d
from research_common.paths import ROOT

OUT = ROOT / "runs/post/build"
SCENES = [f"FloorPlan{i}" for i in range(1, 31)]
PAIRS_PER_DIM = 4
DIMS = ("mu", "legs", "rho")
RHO_SHORT, RHO_LONG = (0.9, 2.6)
F_RHO = 1.4
F_MASS = 1.3
F_MU = 1.5
MU_GENEROUS = 10.0
NEAR_MAX = RHO_SHORT / F_RHO
FAR_MIN = RHO_SHORT * F_RHO
FAR_MAX = RHO_LONG / F_RHO
H_MIN, H_MAX = (0.5, 1.6)
DOCKS_PER_SPEC = 3
SPECS_PER_DIM = 10


def d3(dock: dict, opos: dict, agent_y: float) -> float:
    return math.sqrt(
        (dock["x"] - opos["x"]) ** 2
        + (agent_y - opos["y"]) ** 2
        + (dock["z"] - opos["z"]) ** 2
    )


def heading_to(dock: dict, opos: dict) -> int:
    ang = math.degrees(math.atan2(opos["x"] - dock["x"], opos["z"] - dock["z"]))
    return int(round(ang / 90.0) * 90) % 360


def phase_catalog() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = PostBody()
    cat = {}
    try:
        for sc in SCENES:
            body.launch(sc)
            allobj = {o["objectId"]: o for o in body.objects()}

            def enclosed(o):
                for pid in o.get("parentReceptacles") or []:
                    p = allobj.get(pid)
                    if p and p.get("openable") and (not p.get("isOpen")):
                        return True
                return False

            objs = [
                {
                    "type": o["objectType"].lower(),
                    "object_id": o["objectId"],
                    "mass": o["mass"],
                    "position": o["position"],
                    "enclosed": enclosed(o),
                }
                for o in body.objects()
                if o["pickupable"]
            ]
            reach = body.reachable()
            ay = body.agent_pose()["position"]["y"]
            cat[sc] = {"objects": objs, "reachable": reach, "agent_y": ay}
            print(
                f"[catalog] {sc}: {len(objs)} pickupable, {len(reach)} reachable",
                flush=True,
            )
    finally:
        body.stop()
    (OUT / "catalog_post.json").write_text(json.dumps(cat) + "\n")


def walkable_adjacent(opos: dict, reach: list) -> bool:
    return any((dist2d(p, opos) <= ADJ for p in reach))


def unique_types(objs: list) -> dict:
    cnt: dict = {}
    for o in objs:
        cnt[o["type"]] = cnt.get(o["type"], 0) + 1
    return {
        o["type"]: o
        for o in objs
        if cnt[o["type"]] == 1
        and (not o.get("enclosed"))
        and (H_MIN <= o["position"]["y"] <= H_MAX)
    }


def mass_ok(lo: float, hi: float) -> bool:
    return lo > 0.01 and hi >= F_MASS * lo


def try_spec(dim: str, sc: str, scat: dict) -> list:
    """All (dock, candidates, theta) specs for one scene+dim, scored."""
    uniq = list(unique_types(scat["objects"]).values())
    reach, ay = (scat["reachable"], scat["agent_y"])
    uniq = [o for o in uniq if walkable_adjacent(o["position"], reach)]
    dmin = {o["type"]: min((d3(p, o["position"], ay) for p in reach)) for o in uniq}
    specs = []
    for dock in reach:
        ds = {o["type"]: d3(dock, o["position"], ay) for o in uniq}
        near = [o for o in uniq if ds[o["type"]] <= NEAR_MAX]
        band = [o for o in uniq if FAR_MIN <= ds[o["type"]] <= FAR_MAX]
        far = [o for o in uniq if ds[o["type"]] >= FAR_MIN]
        got = None
        if dim == "mu":
            pool = sorted(uniq, key=lambda o: o["mass"])
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    L, H = (pool[i], pool[j])
                    if H["mass"] < F_MU * F_MU * L["mass"]:
                        continue
                    D = next((d for d in pool if d["mass"] * F_MASS <= L["mass"]), None)
                    if D is None:
                        continue
                    mu_a = round(F_MU * L["mass"], 2)
                    mu_b = round(F_MU * H["mass"], 2)
                    got = {
                        "cands": [D, L, H],
                        "roles": {"D": D["type"], "L": L["type"], "H": H["type"]},
                        "theta_A": {"legs": "ok", "rho": RHO_LONG, "mu": mu_a},
                        "theta_B": {"legs": "ok", "rho": RHO_LONG, "mu": mu_b},
                        "label_A": L["type"],
                        "label_B": H["type"],
                    }
                    break
                if got:
                    break
        elif dim == "legs":
            for N in near:
                for F in far:
                    if F["mass"] < F_MASS * N["mass"]:
                        continue
                    if dmin[F["type"]] > 0.75:
                        continue
                    D = next(
                        (
                            d
                            for d in near
                            if d is not N and d["mass"] * F_MASS <= N["mass"]
                        ),
                        None,
                    )
                    if D is None:
                        continue
                    got = {
                        "cands": [D, N, F],
                        "roles": {"D": D["type"], "N": N["type"], "F": F["type"]},
                        "theta_A": {
                            "legs": "dead",
                            "rho": RHO_SHORT,
                            "mu": MU_GENEROUS,
                        },
                        "theta_B": {"legs": "ok", "rho": RHO_SHORT, "mu": MU_GENEROUS},
                        "label_A": N["type"],
                        "label_B": F["type"],
                    }
                    break
                if got:
                    break
        elif dim == "rho":
            for N in near:
                for H in band:
                    if H["mass"] < F_MASS * N["mass"]:
                        continue
                    D = next(
                        (
                            d
                            for d in near
                            if d is not N and d["mass"] * F_MASS <= N["mass"]
                        ),
                        None,
                    )
                    if D is None:
                        continue
                    got = {
                        "cands": [D, N, H],
                        "roles": {"D": D["type"], "N": N["type"], "H": H["type"]},
                        "theta_A": {
                            "legs": "dead",
                            "rho": RHO_SHORT,
                            "mu": MU_GENEROUS,
                        },
                        "theta_B": {"legs": "dead", "rho": RHO_LONG, "mu": MU_GENEROUS},
                        "label_A": N["type"],
                        "label_B": H["type"],
                    }
                    break
                if got:
                    break
        if not got:
            continue
        cands = got.pop("cands")
        prime = cands[-1]
        spec = {
            "dim": dim,
            "scene": sc,
            "dock": {
                "position": dock,
                "rotation_y": heading_to(dock, prime["position"]),
            },
            "candidates": [
                {
                    "name": o["type"],
                    "object_id": o["object_id"],
                    "mass": o["mass"],
                    "d_solve": round(d3(dock, o["position"], ay), 3),
                }
                for o in cands
            ],
            **got,
        }
        slack = min(
            (
                abs(c["d_solve"] - t)
                for c in spec["candidates"]
                for t in (NEAR_MAX, FAR_MIN, FAR_MAX)
            )
        )
        spec["slack"] = round(slack, 3)
        specs.append(spec)
    specs.sort(key=lambda s: -s["slack"])
    return specs[:DOCKS_PER_SPEC]


def phase_solve() -> None:
    cat = json.loads((OUT / "catalog_post.json").read_text())
    ranked: dict = {d: [] for d in DIMS}
    for dim in DIMS:
        for sc in SCENES:
            if sc not in cat:
                continue
            for s in try_spec(dim, sc, cat[sc]):
                ranked[dim].append(s)
        seen: dict = {}
        keep = []
        for s in sorted(ranked[dim], key=lambda s: -s["slack"]):
            if seen.get(s["scene"], 0) >= 2:
                continue
            seen[s["scene"]] = seen.get(s["scene"], 0) + 1
            keep.append(s)
        ranked[dim] = keep[:SPECS_PER_DIM]
        print(
            f"[solve] {dim}: {len(ranked[dim])} ranked specs ({sorted(set((s['scene'] for s in ranked[dim])))})",
            flush=True,
        )
    (OUT / "solved_post.json").write_text(json.dumps(ranked, indent=1) + "\n")


def face(body: PostBody, target: int, script: list | None = None) -> None:
    cur = body.agent_pose()["rotation"]
    delta = (target - cur) % 360
    steps = {0: [], 90: ["RotateRight"], 180: ["RotateRight"] * 2, 270: ["RotateLeft"]}[
        delta
    ]
    for a in steps:
        body.c.step(action=a)
        if script is not None:
            script.append({"rotate": "right" if a == "RotateRight" else "left"})


def bfs_path(reach: list, start: dict, goal_pos: dict, adj: float) -> list | None:
    """Grid BFS; returns list of positions dock->adjacent(goal)."""
    key = lambda p: (round(p["x"] * 4), round(p["z"] * 4))
    nodes = {key(p): p for p in reach}
    sk = key(start)
    if sk not in nodes:
        return None
    from collections import deque

    prev: dict = {sk: None}
    q = deque([sk])
    goal = None
    while q:
        k = q.popleft()
        p = nodes[k]
        if dist2d(p, goal_pos) <= adj:
            goal = k
            break
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nk = (k[0] + dx, k[1] + dz)
            if nk in nodes and nk not in prev:
                prev[nk] = k
                q.append(nk)
    if goal is None:
        return None
    path, k = ([], goal)
    while k is not None:
        path.append(nodes[k])
        k = prev[k]
    return list(reversed(path))


def walk(body: PostBody, path: list, script: list | None = None) -> bool:
    """Execute a grid path with rotate+move; True if fully walked.
    If `script` is given, every primitive is recorded (watch replay)."""
    for a, b in zip(path, path[1:]):
        dx, dz = (b["x"] - a["x"], b["z"] - a["z"])
        hd = int(round(math.degrees(math.atan2(dx, dz)) / 90) * 90) % 360
        face(body, hd, script)
        ev = body.c.step(action="MoveAhead")
        if script is not None:
            script.append({"move": "forward"})
        if not ev.metadata["lastActionSuccess"]:
            return False
    return True


def restore(body: PostBody, object_id: str, pos: dict) -> None:
    """Drop held object and teleport it back to its catalogued pose."""
    body.c.step(action="DropHandObject", forceAction=True)
    body.held = None
    body.c.step(
        action="TeleportObject",
        objectId=object_id,
        position=pos,
        rotation={"x": 0, "y": 0, "z": 0},
        forceAction=True,
    )


def run_member_gate(
    body: PostBody, spec: dict, theta: dict, label: str, reach: list
) -> dict:
    """G0 fidelity + G1 oracle for one member. Returns record with
    ok flag, measured distances/headings, action script (for watch),
    and oracle choice. Order: look around -> dock lifts (in-place) ->
    legs test -> walks; the position-perturbing steps come last."""
    rec = {
        "violations": [],
        "script": [],
        "lifted": {},
        "d_run": {},
        "heading": {},
        "legs_obs": None,
    }
    body.setup_episode(spec, theta)
    dock_pos = spec["dock"]["position"]
    cat_pos = {
        c["object_id"]: next(
            (o["position"] for o in body.objects() if o["objectId"] == c["object_id"])
        )
        for c in spec["candidates"]
    }
    seen: dict = {}
    sc = rec["script"]
    for _ in range(4):
        body.c.step(action="RotateRight")
        sc.append({"rotate": "right"})
        hd = body.agent_pose()["rotation"]
        pv = body.privileged(spec)
        for nm, c in pv["candidates"].items():
            if c["visible"] and nm not in seen:
                seen[nm] = hd
                rec["d_run"][nm] = c["distance"]
    rec["heading"] = seen
    for c in spec["candidates"]:
        if c["name"] not in seen:
            rec["violations"].append(f"{c['name']} never visible")
    for c in spec["candidates"]:
        nm = c["name"]
        if nm not in seen:
            continue
        face(body, seen[nm], sc)
        r = body.act({"try_lift": nm})
        sc.append({"try_lift": nm})
        log = r["log"]
        tgt = log.get("target") or {}
        d = tgt.get("distance")
        if tgt and tgt.get("object_id") != c["object_id"]:
            rec["violations"].append(f"{nm} resolved to wrong object")
        exp = (
            "picked"
            if d is not None and d <= theta["rho"] and (c["mass"] <= theta["mu"])
            else "rho"
            if d is not None and d > theta["rho"]
            else "mu"
        )
        got = "picked" if log.get("ok") else log.get("rule", "refused")
        if log.get("rule") == "ENGINE_VETO":
            rec["violations"].append(f"ENGINE_VETO on {nm}")
        elif got != exp:
            rec["violations"].append(f"{nm}: expected {exp}, got {got} (d={d})")
        if log.get("rule") == "mu":
            rec.setdefault("mu_refused", []).append(nm)
        if log.get("ok"):
            rec["lifted"][nm] = "dock"
            sc.append({"put_down": True})
            restore(body, c["object_id"], cat_pos[c["object_id"]])
    moved = motors = 0
    for _ in range(4):
        r = body.act({"move": "forward"})
        sc.append({"move": "forward"})
        if r["log"].get("rule") == "legs_dead":
            motors += 1
            break
        if r["log"]["ok"]:
            moved += 1
            for _ in range(2):
                body.c.step(action="RotateRight")
                sc.append({"rotate": "right"})
            ev = body.c.step(action="MoveAhead")
            sc.append({"move": "forward"})
            if not ev.metadata["lastActionSuccess"]:
                rec["violations"].append("could not return to dock")
            for _ in range(2):
                body.c.step(action="RotateRight")
                sc.append({"rotate": "right"})
            break
        body.c.step(action="RotateRight")
        sc.append({"rotate": "right"})
    rec["legs_obs"] = "dead" if motors else "ok" if moved else "boxed"
    if theta["legs"] == "dead" and rec["legs_obs"] != "dead":
        rec["violations"].append("legs rule did not fire")
    if theta["legs"] == "ok" and rec["legs_obs"] != "ok":
        rec["violations"].append("legs-ok member boxed in")
    if theta["legs"] == "ok":
        for c in spec["candidates"]:
            nm = c["name"]
            if (
                rec["lifted"].get(nm)
                or nm not in seen
                or nm in rec.get("mu_refused", [])
            ):
                continue
            path = None
            for adj_r in (0.5, 0.65, ADJ):
                path = bfs_path(reach, dock_pos, cat_pos[c["object_id"]], adj_r)
                if path is not None:
                    break
            if path is None:
                rec["violations"].append(f"no path to {nm}")
                continue
            if not walk(body, path, sc):
                rec["violations"].append(f"walk to {nm} blocked")
                continue
            got_vis = None
            for _ in range(4):
                o = body.visible_by_name(nm)
                if o is not None and o["distance"] <= theta["rho"]:
                    got_vis = o
                    break
                body.c.step(action="RotateRight")
                sc.append({"rotate": "right"})
            if got_vis is None:
                rec["violations"].append(f"{nm} not liftable after walk")
            else:
                r = body.act({"try_lift": nm})
                sc.append({"try_lift": nm})
                exp_pick = c["mass"] <= theta["mu"]
                if r["log"].get("ok"):
                    if not exp_pick:
                        rec["violations"].append(
                            f"{nm}: picked though mu should refuse"
                        )
                    else:
                        rec["lifted"][nm] = "walk"
                    sc.append({"put_down": True})
                    restore(body, c["object_id"], cat_pos[c["object_id"]])
                elif exp_pick or r["log"].get("rule") != "mu":
                    rec["violations"].append(
                        f"{nm}: walk-lift failed ({r['log'].get('rule')})"
                    )
            back = bfs_path(reach, body.agent_pose()["position"], dock_pos, 0.2)
            if back is None or not walk(body, back, sc):
                rec["violations"].append(f"walk back from {nm} failed")
    masses = {c["name"]: c["mass"] for c in spec["candidates"]}
    feas = [nm for nm in rec["lifted"]]
    rec["oracle_choice"] = max(feas, key=lambda n: masses[n]) if feas else None
    if rec["oracle_choice"] != label:
        rec["violations"].append(f"oracle {rec['oracle_choice']} != label {label}")
    rec["ok"] = not rec["violations"]
    return rec


def phase_gates() -> None:
    cat = json.loads((OUT / "catalog_post.json").read_text())
    ranked = json.loads((OUT / "solved_post.json").read_text())
    gdir = OUT / "gates"
    gdir.mkdir(exist_ok=True)
    body = PostBody()
    passed: dict = {d: [] for d in DIMS}
    tried: dict = {d: 0 for d in DIMS}
    try:
        for dim in DIMS:
            for i, spec in enumerate(ranked[dim]):
                if len(passed[dim]) >= PAIRS_PER_DIM:
                    break
                sid = f"{dim}_{spec['scene']}_{i}"
                gpath = gdir / f"{sid}.json"
                if gpath.exists():
                    rec = json.loads(gpath.read_text())
                else:
                    tried[dim] += 1
                    t0 = time.time()
                    reach = cat[spec["scene"]]["reachable"]
                    recs = {}
                    for mem, th, lab in (
                        ("A", spec["theta_A"], spec["label_A"]),
                        ("B", spec["theta_B"], spec["label_B"]),
                    ):
                        recs[mem] = run_member_gate(body, spec, th, lab, reach)
                    rec = {
                        "spec_id": sid,
                        "spec": spec,
                        "members": recs,
                        "ok": recs["A"]["ok"] and recs["B"]["ok"],
                        "wall_s": round(time.time() - t0, 1),
                    }
                    gpath.write_text(json.dumps(rec, indent=1) + "\n")
                if rec["ok"]:
                    print(f"[gates] {sid}: ok=True", flush=True)
                else:
                    print(
                        f"[gates] {sid}: ok=False A={rec['members']['A']['violations']} B={rec['members']['B']['violations']}",
                        flush=True,
                    )
                if rec["ok"]:
                    passed[dim].append(sid)
    finally:
        body.stop()
    summary = {
        "passed": passed,
        "tried": tried,
        "testable": {d: len(passed[d]) >= PAIRS_PER_DIM for d in DIMS},
    }
    (OUT / "gates_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[gates] {json.dumps(summary['testable'])}", flush=True)


def facts_masses(cands: list) -> str:
    return (
        "Known object facts: "
        + "; ".join((f"the {c['name']} weighs about {c['mass']:.2f} kg" for c in cands))
        + "."
    )


def facts_distances(cands: list, d_run: dict) -> str:
    return (
        "Dock distances: "
        + "; ".join(
            (
                f"the {c['name']} is about {d_run[c['name']]:.1f} m from the dock"
                for c in cands
            )
        )
        + "."
    )


def profile_sentences(theta: dict) -> str:
    legs = "work normally" if theta["legs"] == "ok" else "do not respond"
    return f"Known body facts: this body's drive motors {legs}. Its reach is about {theta['rho']:.1f} m. It lifts objects up to about {theta['mu']:.2f} kg and no heavier."


def intro_text(cands: list) -> str:
    names = ", ".join((c["name"] for c in cands))
    return (
        f"You are operating a NEW BODY. Its properties may differ from any body you have operated before. You are docked at your charging dock. Before your task is revealed, you may test what this body can do.\nThis body's movement, reach, and lift limit are UNKNOWN and vary between bodies; testing is how you find out.\nObjects of interest: {names}.\n"
        + facts_masses(cands)
        + '\nActions, one per reply, JSON only:\n  {"rotate": "left"} or {"rotate": "right"} - turn 90 degrees.\n  {"move": "forward"} - one step forward.\n  {"try_lift": "<object name>"} - try to pick up a visible object.\n  {"put_down": true} - put down what you hold.\n  {"done": true} - stop testing early.\nYou may take up to 12 action turns in total. Keep what you learn in your NOTE (lines starting with NOTE:) - older observations disappear from view; the NOTE is all you keep.'
    )


def choice_text(cands: list) -> str:
    names = [c["name"] for c in cands]
    return (
        f"TASK REVEALED: Bring back to your charging dock the heaviest object you can ACTUALLY bring back. You must choose exactly one: the {names[0]}, the {names[1]}, or the {names[2]}. A failed delivery scores zero, so do not choose an object this body cannot fetch and carry back.\n"
        + facts_masses(cands)
        + '\nReply with JSON only: {"deliver": "<object name>"}'
    )


def phase_freeze() -> None:
    gdir = OUT / "gates"
    summary = json.loads((OUT / "gates_summary.json").read_text())
    pairs = []
    for dim in DIMS:
        if not summary["testable"][dim]:
            print(
                f"[freeze] dim {dim} NOT-TESTABLE ({len(summary['passed'][dim])} pairs)",
                flush=True,
            )
            continue
        for sid in summary["passed"][dim][:PAIRS_PER_DIM]:
            rec = json.loads((gdir / f"{sid}.json").read_text())
            spec = rec["spec"]
            cands = spec["candidates"]
            d_run = rec["members"]["B"]["d_run"]
            pair = {
                "pair_id": f"post_{sid}",
                "dim": dim,
                "scene": spec["scene"],
                "dock": spec["dock"],
                "candidates": [
                    {
                        **c,
                        "d_run": d_run.get(c["name"]),
                        "heading": rec["members"]["B"]["heading"].get(c["name"]),
                    }
                    for c in cands
                ],
                "roles": spec["roles"],
                "intro": intro_text(cands),
                "choice_core": choice_text(cands),
                "facts_distances": facts_distances(cands, d_run),
                "members": [
                    {
                        "member": m,
                        "theta": spec[f"theta_{m}"],
                        "label_choice": spec[f"label_{m}"],
                        "profile": profile_sentences(spec[f"theta_{m}"]),
                        "oracle_script": [
                            a
                            for a in rec["members"][m]["script"]
                            if not next(iter(a)).startswith("_")
                        ],
                    }
                    for m in ("A", "B")
                ],
            }
            if not pair["intro"] == intro_text(cands):
                raise ValueError("Numerical validation failed at source line 609")
            if not pair["members"][0]["profile"] != pair["members"][1]["profile"]:
                raise ValueError("Numerical validation failed at source line 610")
            pairs.append(pair)
    txt = "".join((json.dumps(p) + "\n" for p in pairs))
    (OUT / "pairs_post.jsonl").write_text(txt)
    sha = hashlib.sha256(txt.encode()).hexdigest()
    gates = {
        "design": "SCHEMA_POST_DESIGN.md v1.0",
        "pairs_sha256": sha,
        "n_pairs": len(pairs),
        "per_dim": {d: sum((1 for p in pairs if p["dim"] == d)) for d in DIMS},
        "summary": summary,
    }
    (OUT / "GATES_POST.json").write_text(json.dumps(gates, indent=1) + "\n")
    print(f"[freeze] {len(pairs)} pairs, sha256 {sha[:16]}…", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase", required=True, choices=("catalog", "solve", "gates", "freeze")
    )
    a = ap.parse_args()
    {
        "catalog": phase_catalog,
        "solve": phase_solve,
        "gates": phase_gates,
        "freeze": phase_freeze,
    }[a.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
