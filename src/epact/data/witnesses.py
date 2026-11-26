"""Replay semantically reviewed legal routes and record Answer explicitly.

The simulator's evidence proxy is diagnostic. Successful oracle replay does
not itself certify readability; replay images still need visual review.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from research_common.paths import ROOT
from epact.data.io import sha, atomic


def worker(args, spec, reviews):
    from PIL import Image, ImageDraw
    from epact.environment.inspection import InspectionEpisode, InspectionSim
    from epact.environment.evidence import assess
    import numpy as np

    pid = args.pair_id
    review = reviews[pid]
    auditpath = (
        Path(review.get("audit_root", str(args.audit_root)))
        / "pairs"
        / pid
        / "audit.json"
    )
    if sha(auditpath) != review["audit_sha256"]:
        raise RuntimeError("reviewed audit changed")
    audit = json.loads(auditpath.read_text())
    tasks = [
        json.loads(l) for l in (args.data_root / "val.jsonl").read_text().splitlines()
    ]
    members = [t for t in tasks if t["pair_id"] == pid]
    if len(members) != 2 or {t["answer"] for t in members} != {"yes", "no"}:
        raise RuntimeError("pair broken")
    dest = args.out / "pairs" / pid
    dest.mkdir(parents=True, exist_ok=True)
    result = {
        "pair_id": pid,
        "family": members[0]["benchmark_family"],
        "review_sha256": spec["review_sha256"],
        "audit_sha256": sha(auditpath),
        "witness_phase": review["witness_phase"],
        "members": [],
    }
    sim = InspectionSim()
    cache = {}
    try:
        for task in members:
            source = next(
                (m for m in audit["members"] if m["task_id"] == task["task_id"])
            )
            view = source["views"][review["witness_phase"]]
            actions = view["actions"]
            if len(actions) >= 14:
                raise RuntimeError("witness leaves no Answer in model step allowance")
            ep = InspectionEpisode(
                sim, task, ROOT / "data/external/procthor-10k/val.jsonl.gz"
            )
            ep.setup(cache)
            start = dest / (task["task_id"] + "__start.png")
            Image.fromarray(ep.frame()).save(start)
            actual = []
            for action in actions:
                if ep.outcome:
                    break
                ep.act(action["action"], **action["args"])
                s = ep.steps[-1]
                actual.append(
                    {
                        "action": action["action"],
                        "args": action["args"],
                        "ok": bool(s.get("ok")),
                        "error": s.get("error", ""),
                    }
                )
            frame = ep.frame().copy()
            witness = dest / (task["task_id"] + "__witness.png")
            Image.fromarray(frame).save(witness)
            original = Image.open(auditpath.parents[2] / view["image"]).convert("RGB")
            diff = float(
                np.abs(
                    frame.astype(np.int16) - np.array(original).astype(np.int16)
                ).mean()
            )
            evidence = assess(ep)
            if ep.outcome is None:
                ep.act("Answer", answer=task["answer"], confidence=1)
            answered = any((s["action"] == "Answer" for s in ep.steps))
            result["members"].append(
                {
                    "task_id": task["task_id"],
                    "answer": task["answer"],
                    "actual_actions": actual,
                    "actions_match_reviewed_successes": [a["ok"] for a in actual]
                    == [a["ok"] for a in actions],
                    "replayed_all_actions": len(actual) == len(actions),
                    "answer_recorded": answered,
                    "outcome": ep.outcome,
                    "step_log": ep.steps,
                    "evidence_proxy": evidence,
                    "witness_mean_absolute_difference": diff,
                    "observations": {
                        "start": {
                            "image": str(start.resolve()),
                            "image_sha256": sha(start),
                        },
                        "witness": {
                            "image": str(witness.resolve()),
                            "image_sha256": sha(witness),
                        },
                    },
                }
            )
    finally:
        sim.stop()
    result["oracle_completes_both"] = all(
        (
            m["answer_recorded"]
            and m["outcome"].get("correct")
            and m["replayed_all_actions"]
            and m["actions_match_reviewed_successes"]
            for m in result["members"]
        )
    )
    result["semantic_replay_review"] = "pending actual replay images"
    sheet = Image.new("RGB", (1024, 580), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), pid + " | " + members[0]["query"], fill="black")
    for col, m in enumerate(result["members"]):
        draw.text(
            (col * 512 + 8, 35),
            m["answer"] + " | oracle=" + str(result["oracle_completes_both"]),
            fill="black",
        )
        sheet.paste(Image.open(m["observations"]["witness"]["image"]), (col * 512, 60))
    sheet.save(dest / "review.jpg", quality=94)
    atomic(dest / "replay.json", result)
    print(
        json.dumps(
            {"pair": pid, "oracle_completes_both": result["oracle_completes_both"]}
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair-id")
    ap.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = {
        "data_sha256": sha(args.data_root / "val.jsonl"),
        "review_sha256": sha(args.review),
        "scene_dataset_sha256": sha(ROOT / "data/external/procthor-10k/val.jsonl.gz"),
        "sources": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [Path(__file__), *sorted((ROOT / "src/epact").glob("*.py"))]
        },
    }
    reviews = json.loads(args.review.read_text())["pairs"]
    if args.pair_id:
        worker(args, spec, reviews)
        return
    specpath = args.out / "spec.json"
    if specpath.exists() and json.loads(specpath.read_text()) != spec:
        raise SystemExit("Use a new replay directory after source changes")
    atomic(specpath, spec)
    ids = [pid for pid, r in reviews.items() if r["decision"] == "accept"]

    def run(pid):
        if (args.out / "pairs" / pid / "replay.json").exists():
            return {"pair": pid, "returncode": 0}
        with (args.out / (pid + ".log")).open("w") as log:
            p = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--data-root",
                    str(args.data_root),
                    "--audit-root",
                    str(args.audit_root),
                    "--review",
                    str(args.review),
                    "--out",
                    str(args.out),
                    "--pair-id",
                    pid,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = p.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                rc = -1
        row = {"pair": pid, "returncode": rc}
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run, ids))
    reports = [
        json.loads(p.read_text()) for p in (args.out / "pairs").glob("*/replay.json")
    ]
    atomic(
        args.out / "report.json",
        {
            "requested": len(ids),
            "captured": len(reports),
            "oracle_completes_both": sum((r["oracle_completes_both"] for r in reports)),
            "jobs": rows,
            "semantic_replay_review": "pending actual replay images; oracle correctness is not readability",
        },
    )


if __name__ == "__main__":
    main()
