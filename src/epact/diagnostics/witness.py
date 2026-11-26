"""Frozen-route handoff; same-frame QA and ordinary interactive continuation."""

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import re
import time
from collections import Counter
from pathlib import Path
from epact.evaluation.protocol import MAX_STEPS
from epact.runtime.client import ServerLink
from epact.runtime.client import advance_state
from epact.evaluation.policy import ask_model
from epact.evaluation.protocol import policy_prompt
from epact.data.paths import house_dataset, repo_path, resolve_observations
from research_common.paths import ROOT

CHECK = ("C", "D", "L", "C+D", "L+D", "D+door")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def public_action(action):
    return {k: action[k] for k in ("action", "args", "ok", "error")}


def validate_twins(members):
    if len(members) != 2 or {m["answer"] for m in members} != {"yes", "no"}:
        raise ValueError("two opposite-answer members required")
    routes = [[public_action(a) for a in m["actual_actions"]] for m in members]
    if routes[0] != routes[1]:
        raise ValueError("pair-dependent public prefix")
    if any((a["action"] == "Answer" for route in routes for a in route)):
        raise ValueError("answer in prefix")
    if len(routes[0]) >= MAX_STEPS:
        raise ValueError("no remaining model decision")
    return routes


def prepare(manifest_path, dataset_path=None, data_root=None):
    capture = resolve_observations(json.loads(manifest_path.read_text()))
    if not capture.get("visually_reviewed"):
        raise ValueError("reviewed manifest required")
    data = (
        Path(data_root)
        if data_root
        else repo_path(
            capture.get("data_root", "artifacts/balanced_eval_v4_frozen_v1/data")
        )
    ) / "val.jsonl"
    if digest(data) != capture["data_sha256"]:
        raise ValueError("frozen data changed")
    if digest(house_dataset(dataset_path)) != capture["scene_dataset_sha256"]:
        raise ValueError("scene data changed")
    for p, h in capture["sources"].items():
        if digest(ROOT / p) != h:
            raise ValueError(f"frozen environment changed: {p}")
    tasks = {t["task_id"]: t for t in map(json.loads, data.read_text().splitlines())}
    pairs, members = ({}, {})
    for pid in capture["pair_ids"]:
        note = capture["pair_notes"][pid]
        provenance = note["replay"]
        if not provenance["replay_visually_reviewed"]:
            raise ValueError(f"unreviewed replay: {pid}")
        replay_path = repo_path(provenance["replay_path"])
        if digest(replay_path) != provenance["replay_sha256"]:
            raise ValueError(f"changed replay: {pid}")
        if digest(repo_path(provenance["review_path"])) != provenance["review_sha256"]:
            raise ValueError(f"changed review: {pid}")
        replay = json.loads(replay_path.read_text())
        if not replay["oracle_completes_both"]:
            raise ValueError(f"incomplete oracle: {pid}")
        twins = replay["members"]
        routes = validate_twins(twins)
        ids = [m["task_id"] for m in twins]
        expected = {t["task_id"] for t in tasks.values() if t.get("pair_id") == pid}
        if set(ids) != expected:
            raise ValueError(f"wrong replay membership: {pid}")
        if any(
            (
                tasks[ids[0]][k] != tasks[ids[1]][k]
                for k in ("query", "house_index", "start_pose", "split")
            )
        ):
            raise ValueError(f"inconsistent twin context: {pid}")
        pairs[pid] = {"ids": ids, "family": note["family"]}
        for m, route in zip(twins, routes):
            tid = m["task_id"]
            if m["answer"] != tasks[tid]["answer"]:
                raise ValueError(f"replay label mismatch: {tid}")
            view = capture["observations"][tid]["witness"]
            if digest(view["image"]) != view["image_sha256"]:
                raise ValueError(f"changed reviewed image: {tid}")
            members[tid] = {
                "task_id": tid,
                "pair_id": pid,
                "family": note["family"],
                "answer": m["answer"],
                "query": tasks[tid]["query"],
                "prefix": route,
                "witness": view,
                "replay_sha256": provenance["replay_sha256"],
            }
    return {
        "eval_manifest_sha256": digest(manifest_path),
        "pairs": pairs,
        "members": members,
        "data_root": str(data.parent),
        "max_total_decisions": MAX_STEPS,
        "budget": 20,
        "hide_budget": True,
    }


def plan_semantics(plan):
    """Compare task/route/image identity without tying a plan to one checkout path."""
    members = {}
    for tid, member in plan["members"].items():
        members[tid] = {
            k: member[k]
            for k in ("task_id", "pair_id", "family", "answer", "query", "prefix")
        }
        members[tid]["witness_image_sha256"] = member["witness"]["image_sha256"]
    return {
        "pairs": plan["pairs"],
        "members": members,
        **{k: plan[k] for k in ("max_total_decisions", "budget", "hide_budget")},
    }


def summarize(rows, pairs):
    by_id = {r["task_id"]: r for r in rows}
    complete = {p: v for p, v in pairs.items() if all((t in by_id for t in v["ids"]))}
    fields = (
        "baseline_correct",
        "old_witness_correct",
        "fresh_qa_correct",
        "first_answer_correct",
        "scored_correct",
    )

    def scores(selected):
        return {
            f: {
                "correct": sum(
                    (all((by_id[t][f] for t in v["ids"])) for v in selected.values())
                ),
                "pairs": len(selected),
                "accuracy": sum(
                    (all((by_id[t][f] for t in v["ids"])) for v in selected.values())
                )
                / len(selected)
                if selected
                else None,
            }
            for f in fields
        }

    family = {
        f: scores({p: v for p, v in complete.items() if v["family"] == f})
        for f in sorted({v["family"] for v in pairs.values()})
    }
    conditioned = {
        p: v
        for p, v in complete.items()
        if all((by_id[t]["fresh_qa_correct"] for t in v["ids"]))
    }
    fully_complete = len(complete) == len(pairs)
    macro = {
        f: sum((family[c][f]["accuracy"] for c in CHECK)) / len(CHECK)
        if fully_complete
        and all((c in family and family[c][f]["pairs"] for c in CHECK))
        else None
        for f in fields
    }
    return {
        "requested_pairs": len(pairs),
        "completed_pairs": len(complete),
        "missing_or_incomplete_pairs": len(pairs) - len(complete),
        "completed_items": len(rows),
        "pair_scores": scores(complete),
        "check_macro": macro,
        "family": family,
        "fresh_qa_correct_pair_subset": scores(conditioned),
        "item_correct": {f: sum((r[f] for r in rows)) for f in fields},
        "first_answer_items": sum((r["first_answer"] for r in rows)),
        "continued_items": sum((not r["first_answer"] for r in rows)),
        "forced_endings": dict(
            Counter((r["forced_answer"] for r in rows if r["forced_answer"]))
        ),
        "model_decisions": sum((len(r["decided"]) for r in rows)),
        "repeated_action_items": sum((r["repeated_actions"] > 0 for r in rows)),
        "image_mae_max": max((r["witness_image_mae"] for r in rows), default=None),
        "image_changed_items": sum((r["witness_image_mae"] > 0 for r in rows)),
        "interpretation": "Assisted diagnostic, not autonomous acquisition. Fresh-QA subset is post-hoc.",
    }


def replay_prefix(link, member):
    state = link.call(cmd="start", task_id=member["task_id"])
    if (
        state.get("protocol") != "inspection-v4-dev"
        or state.get("query") != member["query"]
    ):
        raise ValueError("server context mismatch")
    history = []
    for expected in member["prefix"]:
        reply = link.call(cmd="act", action=expected["action"], args=expected["args"])
        if reply["terminated"]:
            raise RuntimeError(f"prefix terminated early: {member['task_id']}")
        result = reply["result"]
        if any((result[k] != expected[k] for k in ("ok", "error"))):
            raise RuntimeError(
                f"prefix public outcome changed: {member['task_id']}: {expected}: {result}"
            )
        history.append({**expected, "cost": result["cost"]})
        state = advance_state(state, reply)
    return (state, history)


def fresh_qa(model, processor, image, query):
    import torch
    from epact.diagnostics.readability import PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT.format(query=query)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    raw = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]
    match = re.search('"answer"\\s*:\\s*"(yes|no)"', raw)
    return {"pred": match.group(1) if match else None, "raw_reply": raw}


def finish_score(decided, outcome, forced, answer):
    first = decided[0]
    first_answer = first["action"] == "Answer" and (not first["parse_failed"])
    last = decided[-1]
    valid_answer = (
        last["action"] == "Answer"
        and (not last["parse_failed"])
        and (last["args"].get("answer") in ("yes", "no"))
        and (forced is None)
    )
    return {
        "first_answer": first_answer,
        "first_answer_correct": first_answer and first["args"].get("answer") == answer,
        "scored_correct": valid_answer and bool(outcome.get("correct")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-manifest",
        type=Path,
        default=ROOT / "configs/epact/evaluation/full.json",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument(
        "--dataset",
        type=Path,
        help="ProcTHOR split file; defaults to EPACT_HOUSE_DATASET or assets/procthor-10k/val.jsonl.gz",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        help="Override the task directory in the evaluation manifest",
    )
    ap.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "data/epact/artifacts/benchmark_balanced_v4_v2/8b",
    )
    ap.add_argument("--port", type=int, default=7367)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument(
        "--smoke-only",
        action="store_true",
        help="Replay first pair of each family on CPU; no VLM",
    )
    args = ap.parse_args()
    args.max_new_tokens, args.hide_budget = (96, True)
    plan = prepare(args.eval_manifest, args.dataset, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("handoff plan changed; use a fresh output directory")
    atomic(plan_path, plan)
    print(
        "PLAN "
        + json.dumps(
            {
                "pairs": len(plan["pairs"]),
                "families": dict(
                    Counter((v["family"] for v in plan["pairs"].values()))
                ),
            }
        ),
        flush=True,
    )
    if args.prepare_only:
        return
    from PIL import Image, ImageChops, ImageStat

    if args.smoke_only:
        link = ServerLink(args.port)
        seen, rows = (set(), [])
        for pid, pair in plan["pairs"].items():
            if pair["family"] in seen:
                continue
            seen.add(pair["family"])
            for tid in pair["ids"]:
                state, history = replay_prefix(link, plan["members"][tid])
                rows.append(
                    {
                        "task_id": tid,
                        "family": pair["family"],
                        "prefix_steps": len(history),
                        "frame_sha256": state["frame_sha256"],
                    }
                )
                print("SMOKE " + json.dumps(rows[-1]), flush=True)
        atomic(args.output_dir / "smoke.json", rows)
        return
    source_files = [
        "src/epact/diagnostics/witness.py",
        "src/epact/evaluation/protocol.py",
        "src/epact/evaluation/policy.py",
        "src/epact/runtime/client.py",
        "src/epact/diagnostics/readability.py",
        "src/epact/data/paths.py",
    ]
    baseline_hashes = {}
    for tid in plan["members"]:
        for rel in (f"episodes/{tid}.json", f"probe_rows/{tid}__witness.json"):
            baseline_hashes[rel] = digest(args.baseline_dir / rel)
    config = {
        "plan_sha256": digest(plan_path),
        "sources": {p: digest(ROOT / p) for p in source_files},
        "model_path": str(args.model_path.resolve()),
        "model_config_sha256": digest(args.model_path / "config.json"),
        "baseline_files": baseline_hashes,
        "do_sample": False,
        "max_new_tokens": 96,
        "qa_max_new_tokens": 48,
        "runtime_versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "Pillow")
        },
    }
    conf_path = args.output_dir / "manifest.json"
    if conf_path.exists() and json.loads(conf_path.read_text()) != config:
        raise ValueError("run configuration changed; use a fresh output directory")
    atomic(conf_path, config)
    result_dir = args.output_dir / "episodes"
    result_dir.mkdir(exist_ok=True)
    rows = {p.stem: json.loads(p.read_text()) for p in result_dir.glob("*.json")}
    if set(rows) - set(plan["members"]):
        raise ValueError("unexpected output tasks")
    pending = [tid for tid in plan["members"] if tid not in rows]
    if pending:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading 8B for {len(pending)} handoff episodes", flush=True)
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = (
            AutoModelForImageTextToText.from_pretrained(
                args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to("cuda")
            .eval()
        )
        print("Model loaded", flush=True)
        link = ServerLink(args.port)
        for tid in pending:
            begin = time.monotonic()
            member = plan["members"][tid]
            state, history = replay_prefix(link, member)
            raw_image = base64.b64decode(state["frame_png_b64"])
            img_path = args.output_dir / "handoff_frames" / f"{tid}.png"
            img_path.parent.mkdir(exist_ok=True)
            img_path.write_bytes(raw_image)
            image = Image.open(io.BytesIO(raw_image)).convert("RGB")
            reviewed = Image.open(member["witness"]["image"]).convert("RGB")
            if image.size != reviewed.size:
                raise ValueError("witness image dimensions changed")
            mae = sum(ImageStat.Stat(ImageChops.difference(image, reviewed)).mean) / 3
            qa = fresh_qa(model, processor, image, member["query"])
            handoff = {
                "frame_sha256": state["frame_sha256"],
                "image_sha256": digest(img_path),
                "privileged_evidence_proxy": state.get("privileged_evidence_proxy"),
                "privileged_determinable": state.get("privileged_determinable"),
                "prompt": policy_prompt(state, history, final=False, hide_budget=True),
            }
            decided, forced, outcome = ([], None, None)
            for _ in range(MAX_STEPS - len(member["prefix"])):
                name, kw = ask_model(
                    model, processor, state, history, args, final=False
                )
                step = {
                    "action": name,
                    "args": kw,
                    "parse_failed": args.last_parse_failed,
                    "raw_reply": args.last_model_reply,
                    "frame_sha256_before": state["frame_sha256"],
                }
                decided.append(step)
                if args.last_parse_failed:
                    forced = "parse_failure"
                reply = link.call(cmd="act", action=name, args=kw)
                if reply["terminated"]:
                    outcome = reply["outcome"]
                    break
                step["result"] = reply["result"]
                history.append({"action": name, "args": kw, **reply["result"]})
                state = advance_state(state, reply)
            if outcome is None:
                forced = "step_limit"
                outcome = link.call(
                    cmd="act", action="Answer", args={"answer": "no", "confidence": 0.0}
                )["outcome"]
            keys = [
                json.dumps({"action": d["action"], "args": d["args"]}, sort_keys=True)
                for d in decided
                if d["action"] != "Answer"
            ]
            baseline = json.loads(
                (args.baseline_dir / "episodes" / f"{tid}.json").read_text()
            )
            old_qa = json.loads(
                (args.baseline_dir / "probe_rows" / f"{tid}__witness.json").read_text()
            )
            row = {
                "task_id": tid,
                "pair_id": member["pair_id"],
                "family": member["family"],
                "answer": member["answer"],
                "prefix": member["prefix"],
                "handoff": handoff,
                "witness_image_mae": mae,
                "fresh_qa": qa,
                "fresh_qa_correct": qa["pred"] == member["answer"],
                "baseline_correct": baseline["scored_correct"],
                "old_witness_correct": old_qa["pred"] == member["answer"],
                "decided": decided,
                "outcome": outcome,
                "forced_answer": forced,
                "repeated_actions": len(keys) - len(set(keys)),
                **finish_score(decided, outcome, forced, member["answer"]),
                "seconds": round(time.monotonic() - begin, 3),
            }
            atomic(result_dir / f"{tid}.json", row)
            rows[tid] = row
            atomic(
                args.output_dir / "summary.json",
                summarize(list(rows.values()), plan["pairs"]),
            )
            print(
                f"[{len(rows)}/{len(plan['members'])}] {tid}: qa={row['fresh_qa_correct']} first={row['first_answer']} correct={row['scored_correct']} decisions={len(decided)} image_mae={mae:.4f}",
                flush=True,
            )
        link.file.close()
        link.sock.close()
    atomic(
        args.output_dir / "summary.json", summarize(list(rows.values()), plan["pairs"])
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
