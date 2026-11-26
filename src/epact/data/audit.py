"""Audit candidate twin routes and capture review images.

Container routes include pre-open views and nested-drawer opening. Route
construction is label independent; saved images require visual review."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from research_common.paths import ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def worker(args, tasks):
    import numpy as np
    from PIL import Image, ImageDraw
    from epact.environment.inspection import InspectionEpisode, InspectionSim
    from epact.environment.evidence import assess

    members = [t for t in tasks if t.get("pair_id") == args.pair_id]
    if len(members) != 2:
        raise RuntimeError("not a pair")
    fields = ["query", "house_index", "start_pose", "split", "benchmark_family"]
    if any((members[0][k] != members[1][k] for k in fields)) or {
        t["answer"] for t in members
    } != {"yes", "no"}:
        raise RuntimeError("pair definition mismatch")
    pairdir = args.out / "pairs" / args.pair_id
    pairdir.mkdir(parents=True, exist_ok=True)
    report = {
        "pair_id": args.pair_id,
        "family": members[0]["benchmark_family"],
        "condition": members[0]["benchmark_condition"],
        "status": "review_pending",
        "members": [],
        "policy": "pixel/geometry/route failures are diagnostics, not task invalidity",
    }
    sim = InspectionSim()
    cache = {}
    starts = []
    try:
        for task in members:
            repeat = []
            for _ in range(2):
                ep = InspectionEpisode(sim, task, args.dataset)
                ep.setup(cache)
                repeat.append(ep.frame().astype(np.int16).copy())
            starts.append(repeat)
            objects = sim.objects()
            by_id = {o["objectId"]: o for o in objects}
            anchor = (
                task.get("receptacle_id")
                if task["query_type"] == "CHECK"
                else task["target_object_id"]
            )
            if task["query_type"] == "CHECK":
                found = [
                    o["objectId"]
                    for o in objects
                    if o["objectType"] == task["target_type"]
                    and anchor in (o.get("parentReceptacles") or [])
                ]
                truthful = bool(found) == (task["answer"] == "yes")
            else:
                found = []
                obj = by_id.get(anchor)
                truthful = (
                    bool(obj) and obj.get(task["state_field"]) == task["state_value"]
                )
            row = {
                "task_id": task["task_id"],
                "query": task["query"],
                "answer": task["answer"],
                "label_matches_live_state": truthful,
                "same_type_in_region": found,
                "views": {},
                "actions": [],
            }
            if not truthful:
                report["status"] = "invalid"

            def snap(stage):
                image = ep.frame()
                path = pairdir / (task["task_id"] + "__" + stage + ".png")
                Image.fromarray(image).save(path)
                evidence = assess(ep)
                row["views"][stage] = {
                    "image": str(path.relative_to(args.out)),
                    "image_sha256": sha(path),
                    "evidence_proxy": evidence,
                    "darkness_active": ep.darkness_active(),
                    "mean_brightness": float(image.mean()),
                    "actions": list(row["actions"]),
                    "public_query_object_reference": ep.adapter.resolve_target(
                        "query_object"
                    ),
                    "outcome": ep.outcome,
                }

            def act(name, **kw):
                if ep.outcome:
                    return False
                out = ep.act(name, **kw)
                s = ep.steps[-1]
                row["actions"].append(
                    {
                        "action": name,
                        "args": kw,
                        "ok": bool(s.get("ok")),
                        "error": s.get("error", ""),
                    }
                )
                return bool(s.get("ok"))

            snap("start")
            reference = (
                "query_location" if task["query_type"] == "CHECK" else "query_object"
            )
            if "L" in task["hiding_kinds"]:
                act("GoTo", target=reference)
                act("LookAt", target=reference)
                snap("without_toggle")
                ep = InspectionEpisode(sim, task, args.dataset)
                ep.setup(cache)
                row["actions"] = []
            for door in (task.get("house_mod") or {}).get("closed_doors") or []:
                act("GoTo", target=door)
                opened = act("Open", target=door)
                for side in ["east", "south", "west", "north"]:
                    if opened or ep.outcome:
                        break
                    act("GoTo", target=door, side=side)
                    opened = act("Open", target=door)
            if "L" in task["hiding_kinds"]:
                act("GoTo", target="room_light")
                act("Toggle", target="room_light")
                snap("after_light")
            act("GoTo", target=reference)
            act("LookAt", target=reference)
            snap("pre_open")
            for item in task.get("oracle_plan", []):
                if task["query_type"] == "CHECK" and item["action"] == "Open":
                    target = item.get("target")
                    obj = sim.get_object(target)
                    if obj and obj.get("openable") and (not obj.get("isOpen")):
                        act("GoTo", target=target)
                        act("Open", target=target)
            obj = sim.get_object(anchor)
            if (
                task["query_type"] == "CHECK"
                and obj
                and obj.get("openable")
                and (not obj.get("isOpen"))
            ):
                act("Open", target=reference)
            act("LookAt", target=reference)
            snap("near")
            if task["query_type"] == "CHECK" and obj:
                children = [
                    o
                    for o in sim.objects()
                    if anchor in (o.get("parentReceptacles") or [])
                    and o.get("openable")
                    and (not o.get("isOpen"))
                ]
                for child in sorted(children, key=lambda o: o["objectId"])[:4]:
                    act("Open", target=child["objectId"])
                if children:
                    act("LookAt", target=reference)
                    snap("deep")
            for stage, view, side, stance in [
                ("wide", "wide", "auto", "stand"),
                ("crouch", "inspect", "auto", "crouch"),
                ("east", "inspect", "east", "stand"),
                ("south", "inspect", "south", "stand"),
            ]:
                if ep.outcome:
                    break
                act("GoTo", target=reference, view=view, side=side, stance=stance)
                snap(stage)
            report["members"].append(row)
    finally:
        sim.stop()
    diff = lambda a, b: float(np.abs(a - b).mean())
    within = [diff(s[0], s[1]) for s in starts]
    cross = [diff(a, b) for a in starts[0] for b in starts[1]]
    report["initial_render"] = {
        "within_max": max(within),
        "between_min": min(cross),
        "needs_leak_review": report["condition"] == "hidden"
        and min(cross) > max(0.05, 1.5 * max(within)),
        "note": "A review flag, not proof of leakage; low-light rendering must be checked behaviorally.",
    }
    report["initial_query_reference_equal"] = (
        report["members"][0]["views"]["start"]["public_query_object_reference"]
        == report["members"][1]["views"]["start"]["public_query_object_reference"]
    )
    phases = list(dict.fromkeys((k for m in report["members"] for k in m["views"])))
    sheet = Image.new("RGB", (1024, 40 + 540 * len(phases)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        report["family"] + " | " + args.pair_id + " | " + members[0]["query"],
        fill="black",
    )
    for col, row in enumerate(report["members"]):
        for index, phase in enumerate(phases):
            view = row["views"].get(phase)
            if not view:
                continue
            y = 40 + index * 540
            draw.text(
                (col * 512 + 8, y),
                row["answer"]
                + " | "
                + phase
                + " | dark="
                + str(view["darkness_active"]),
                fill="black",
            )
            sheet.paste(
                Image.open(args.out / view["image"]).convert("RGB"), (col * 512, y + 20)
            )
    sheet.save(pairdir / "review.jpg", quality=93)
    atomic(pairdir / "audit.json", report)
    print(
        json.dumps(
            {
                "pair_id": args.pair_id,
                "status": report["status"],
                "family": report["family"],
                "render": report["initial_render"],
            }
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/external/procthor-10k/train.jsonl.gz",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair-id")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--workers", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()
    tasks = [
        json.loads(l) for l in (args.data_root / "train.jsonl").read_text().splitlines()
    ]
    if args.pair_id:
        worker(args, tasks)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "data_sha256": sha(args.data_root / "train.jsonl"),
        "sources": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__),
                ROOT / "src/epact/environment/pairing.py",
                ROOT / "src/epact/environment/inspection.py",
                ROOT / "src/epact/environment/episode.py",
                ROOT / "src/epact/environment/evidence.py",
            ]
        },
    }
    spec = args.out / "spec.json"
    if spec.exists() and json.loads(spec.read_text()) != config:
        raise SystemExit("Use a new audit directory after changes")
    atomic(spec, config)
    families = defaultdict(list)
    for t in tasks:
        if t["pair_id"] not in families[t["benchmark_family"]]:
            families[t["benchmark_family"]].append(t["pair_id"])
    ids = [
        pids[i]
        for i in range(max(map(len, families.values())))
        for pids in families.values()
        if i < len(pids)
    ]
    failures = []

    def capture_pair(pid):
        dest = args.out / "pairs" / pid / "audit.json"
        if dest.exists():
            return None
        log = args.out / (pid + ".log")
        with log.open("w") as stream:
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "epact.data.audit",
                    "--data-root",
                    str(args.data_root),
                    "--out",
                    str(args.out),
                    "--pair-id",
                    pid,
                    "--dataset",
                    str(args.dataset),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = p.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                rc = -1
        print(json.dumps({"pair": pid, "returncode": rc}), flush=True)
        if rc or not dest.exists():
            return {
                "pair_id": pid,
                "status": "unresolved",
                "returncode": rc,
                "log": str(log),
            }
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(capture_pair, pid) for pid in ids]
        for future in as_completed(futures):
            failure = future.result()
            if failure:
                failures.append(failure)
    reports = [
        json.loads(p.read_text()) for p in (args.out / "pairs").glob("*/audit.json")
    ]
    atomic(
        args.out / "report.json",
        {
            "requested_pairs": len(ids),
            "captured_pairs": len(reports),
            "statuses": dict(Counter((r["status"] for r in reports))),
            "families": dict(Counter((r["family"] for r in reports))),
            "unresolved_runtime": failures,
            "semantic_acceptance": "pending actual image review; no proxy auto-exclusion",
        },
    )


if __name__ == "__main__":
    main()
