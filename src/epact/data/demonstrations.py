"""Capture SFT demonstrations from reviewed witness selections.

Rows retain the actual frame, public history, prompt and target action;
use epact data demonstrations --help for explicit file inputs."""

import argparse
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path
from epact.runtime.client import ServerLink
from epact.runtime.client import advance_state
from epact.evaluation.protocol import policy_prompt
from research_common.paths import ROOT

OUT = ROOT / "runs/epact/demonstrations"


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7400)
    ap.add_argument(
        "--selection",
        type=Path,
        default=ROOT
        / "data/epact/artifacts/balanced_train_pool_v1/training_selection_v1.json",
    )
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N pairs")
    args = ap.parse_args()
    OUT = args.output_dir
    sel = json.loads(args.selection.read_text())
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    (OUT / "members").mkdir(exist_ok=True)
    pairs = sorted(sel["selected"].items())
    if args.limit:
        pairs = pairs[: args.limit]
    link = ServerLink(args.port)
    counts = Counter()
    for pid, pair in pairs:
        routes = [
            [
                {"action": a["action"], "args": a["args"]}
                for a in m["actions"]
                if a["ok"]
            ]
            for m in pair["members"]
        ]
        symmetric = routes[0] == routes[1]
        for m, route in zip(pair["members"], routes):
            done = OUT / "members" / (m["task_id"] + ".json")
            if done.exists():
                counts["member_cached"] += 1
                continue
            state = link.call(cmd="start", task_id=m["task_id"])
            if state.get("protocol") != "inspection-v4-dev":
                raise SystemExit("wrong server protocol")
            history, rows, failed = ([], [], None)
            for step in route:
                prompt = policy_prompt(state, history, final=False, hide_budget=True)
                png = base64.b64decode(state["frame_png_b64"])
                fh = hashlib.sha256(png).hexdigest()
                (OUT / "frames" / f"{fh}.png").write_bytes(png)
                rows.append(
                    {
                        "task_id": m["task_id"],
                        "pair_id": pid,
                        "family": pair["family"],
                        "kind": "action",
                        "query": state["query"],
                        "prompt": prompt,
                        "frame_sha256": fh,
                        "action": step,
                        "history": list(history),
                    }
                )
                reply = link.call(cmd="act", action=step["action"], args=step["args"])
                if reply["terminated"] or not reply["result"]["ok"]:
                    failed = f"replay step failed: {step} -> {(reply['result'].get('error') if not reply['terminated'] else 'terminated')}"
                    break
                history.append(
                    {
                        "action": step["action"],
                        "args": step["args"],
                        **{k: reply["result"][k] for k in ("ok", "error", "cost")},
                    }
                )
                state = advance_state(state, reply)
            if failed:
                done.write_text(json.dumps({"status": "failed", "reason": failed}))
                counts["member_failed"] += 1
                print(
                    json.dumps(
                        {"member": m["task_id"], "status": "failed", "reason": failed}
                    ),
                    flush=True,
                )
                continue
            prompt = policy_prompt(state, history, final=False, hide_budget=True)
            png = base64.b64decode(state["frame_png_b64"])
            fh = hashlib.sha256(png).hexdigest()
            (OUT / "frames" / f"{fh}.png").write_bytes(png)
            answer_row = {
                "task_id": m["task_id"],
                "pair_id": pid,
                "family": pair["family"],
                "kind": "answer",
                "query": state["query"],
                "prompt": prompt,
                "frame_sha256": fh,
                "action": {
                    "action": "Answer",
                    "args": {"answer": m["answer"], "confidence": 0.9},
                },
                "history": list(history),
            }
            record = {
                "status": "ok",
                "symmetric_prefix": symmetric,
                "witness_sha256_expected": m["image_sha256"],
                "witness_sha256_replayed": fh,
                "action_rows": rows,
                "answer_row": answer_row,
            }
            done.write_text(json.dumps(record))
            counts["member_ok"] += 1
            counts["symmetric" if symmetric else "asymmetric"] += 1
            print(
                json.dumps(
                    {
                        "member": m["task_id"],
                        "status": "ok",
                        "steps": len(rows),
                        "symmetric": symmetric,
                    }
                ),
                flush=True,
            )
    link.file.close()
    link.sock.close()
    members = {
        p.stem: json.loads(p.read_text()) for p in (OUT / "members").glob("*.json")
    }
    by_pair = {}
    for pid, pair in sorted(sel["selected"].items()):
        recs = [members.get(m["task_id"]) for m in pair["members"]]
        if not all((r and r["status"] == "ok" for r in recs)):
            continue
        by_pair[pid] = recs
    answer_rows, traj_rows = ([], [])
    for pid, recs in by_pair.items():
        for r in recs:
            answer_rows.append(r["answer_row"])
        if all((r["symmetric_prefix"] for r in recs)):
            for r in recs:
                traj_rows.extend(r["action_rows"])
    traj_rows = traj_rows + answer_rows
    for name, rows in (
        ("answer_rows.jsonl", answer_rows),
        ("trajectory_rows.jsonl", traj_rows),
    ):
        tmp = OUT / (name + ".tmp")
        tmp.write_text("".join((json.dumps(r) + "\n" for r in rows)))
        tmp.replace(OUT / name)
    report = {
        "pairs_selected": len(sel["selected"]),
        "pairs_complete": len(by_pair),
        "answer_rows": len(answer_rows),
        "trajectory_rows_total": len(traj_rows),
        "trajectory_action_rows": len(traj_rows) - len(answer_rows),
        "asymmetric_prefix_pairs_answer_only": sum(
            (
                1
                for recs in by_pair.values()
                if not all((r["symmetric_prefix"] for r in recs))
            )
        ),
        "member_counts": dict(counts),
        "note": "Answer rows byte-identical between files (matched exposure). Prompts captured live from inspection-v4-dev, budget hidden, final=False.",
    }
    (OUT / "rows_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
