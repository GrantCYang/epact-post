"""Fixed public prompt at original start and reviewed witness handoff."""

import argparse
import base64
import importlib.metadata
import io
import json
import time
from collections import Counter
from pathlib import Path
from epact.evaluation.protocol import MAX_STEPS
from epact.runtime.client import ServerLink
from epact.runtime.client import advance_state
from epact.evaluation.protocol import parse_action
from epact.evaluation.protocol import policy_prompt
from epact.diagnostics.witness import (
    ROOT,
    CHECK,
    atomic,
    digest,
    prepare,
    replay_prefix,
    fresh_qa,
    finish_score,
    plan_semantics,
)

PROCEDURE = "\nDecision procedure:\nUse the current image and the available action-result history to choose the next step.\nIf the evidence is sufficient to support yes or no, use Answer now.\nIf evidence is insufficient, choose an action that can obtain the missing information.\nNot seeing the target does not by itself establish absence; inspect the relevant area sufficiently before concluding that the target is absent.\nIf an action has already been performed without providing new information, avoid repeating the same action without a reason. Reconsider whether to change the inspection or answer using the evidence already available.\nReply with exactly one action JSON in the specified format."
OUT = ROOT / "runs/epact/procedure"
BASE = ROOT / "data/epact/artifacts/benchmark_balanced_v4_v2/8b"
HANDOFF = ROOT / "data/epact/artifacts/witness_handoff_v1_8b"
ARMS = ("start", "handoff")


def procedure_prompt(state, history):
    return policy_prompt(state, history, final=False, hide_budget=True) + PROCEDURE


def decision_order(plan):
    return [
        (arm, tid)
        for pair in plan["pairs"].values()
        for arm in ARMS
        for tid in pair["ids"]
    ]


def initial_state(link, member, arm):
    if arm not in ARMS:
        raise ValueError("unknown start condition")
    return replay_prefix(link, member if arm == "handoff" else {**member, "prefix": []})


def action_stats(decided):
    keys = [
        json.dumps({"action": d["action"], "args": d["args"]}, sort_keys=True)
        for d in decided
        if d["action"] != "Answer"
    ]
    return {
        "repeated_actions": len(keys) - len(set(keys)),
        "action_counts": dict(Counter((d["action"] for d in decided))),
    }


def cell_summary(rows, pairs):
    by_id = {r["task_id"]: r for r in rows}
    complete = {
        pid: p for pid, p in pairs.items() if all((t in by_id for t in p["ids"]))
    }
    family = {}
    for f in sorted({p["family"] for p in pairs.values()}):
        group = [p for p in complete.values() if p["family"] == f]
        correct = sum(
            (all((by_id[t]["scored_correct"] for t in p["ids"])) for p in group)
        )
        family[f] = {
            "pairs": len(group),
            "correct": correct,
            "accuracy": correct / len(group) if group else None,
        }
    done = len(complete) == len(pairs)
    correct = sum(
        (all((by_id[t]["scored_correct"] for t in p["ids"])) for p in complete.values())
    )
    macro = (
        sum((family[f]["accuracy"] for f in CHECK)) / len(CHECK)
        if done and all((f in family and family[f]["pairs"] for f in CHECK))
        else None
    )
    return {
        "completed_items": len(rows),
        "requested_pairs": len(pairs),
        "completed_pairs": len(complete),
        "missing_pairs": len(pairs) - len(complete),
        "pair_correct": correct,
        "pair_accuracy": correct / len(complete) if complete else None,
        "check_macro": macro,
        "family": family,
        "item_correct": sum((r["scored_correct"] for r in rows)),
        "first_answer_items": sum((r["first_answer"] for r in rows)),
        "repeated_action_items": sum((r["repeated_actions"] > 0 for r in rows)),
        "forced_endings": dict(
            Counter((r["forced_answer"] for r in rows if r["forced_answer"]))
        ),
        "action_counts": dict(
            sum((Counter(r["action_counts"]) for r in rows), Counter())
        ),
        "model_decisions": sum((len(r["decided"]) for r in rows)),
        "input_tokens": sum(
            (d.get("input_tokens", 0) for r in rows for d in r["decided"])
        ),
        "output_tokens": sum(
            (d.get("output_tokens", 0) for r in rows for d in r["decided"])
        ),
        "token_counts_recorded": all(
            ("input_tokens" in d for r in rows for d in r["decided"])
        )
        if rows
        else False,
    }


def historical(plan, base=BASE, handoff=HANDOFF):
    result = {}
    for arm, folder in [("start", base), ("handoff", handoff)]:
        rows = []
        for tid in plan["members"]:
            r = json.loads((folder / "episodes" / f"{tid}.json").read_text())
            d = r["decided"]
            first = d[0]
            rows.append(
                {
                    **r,
                    "task_id": tid,
                    **action_stats(d),
                    "first_answer": first["action"] == "Answer"
                    and (not first.get("parse_failed", False)),
                }
            )
        result[arm] = rows
    return result


def refresh_report(out, plan, old):
    current = {
        arm: [
            json.loads(p.read_text()) for p in (out / arm / "episodes").glob("*.json")
        ]
        for arm in ARMS
    }
    for arm, rows in current.items():
        if len({r["task_id"] for r in rows}) != len(rows) or any(
            (r["task_id"] not in plan["members"] or r["arm"] != arm for r in rows)
        ):
            raise ValueError("unexpected or duplicate output task")
    cells = {
        f"{arm}_{prompt}": cell_summary(rows, plan["pairs"])
        for arm in ARMS
        for prompt, rows in [("original", old[arm]), ("procedure", current[arm])]
    }
    qa_pairs = {
        pid: p
        for pid, p in plan["pairs"].items()
        if all(
            (
                next((r for r in old["handoff"] if r["task_id"] == t))[
                    "fresh_qa_correct"
                ]
                for t in p["ids"]
            )
        )
    }
    qa_ids = {t for p in qa_pairs.values() for t in p["ids"]}
    conditional = {
        name: cell_summary([r for r in rows if r["task_id"] in qa_ids], qa_pairs)
        for name, rows in [
            ("original", old["handoff"]),
            ("procedure", current["handoff"]),
        ]
    }
    new_qa_rows = [
        {**r, "scored_correct": r["fresh_qa"]["pred"] == r["answer"]}
        for r in current["handoff"]
    ]
    qa_summary = cell_summary(new_qa_rows, plan["pairs"])
    done = all((cells[f"{arm}_procedure"]["missing_pairs"] == 0 for arm in ARMS))
    gains = {
        arm: cells[f"{arm}_procedure"]["check_macro"]
        - cells[f"{arm}_original"]["check_macro"]
        if cells[f"{arm}_procedure"]["check_macro"] is not None
        else None
        for arm in ARMS
    }
    summary = {
        "complete": done,
        "cells": cells,
        "check_macro_prompt_gain": gains,
        "fixed_previous_44_qa_pair_subset": conditional,
        "fresh_handoff_qa": qa_summary,
        "completed_new_episodes": sum((len(v) for v in current.values())),
    }
    atomic(out / "comparison.json", summary)
    lines = [
        "# EPACT fixed public procedure prompt",
        "",
        f"Status: {('complete' if done else 'running; partial results')}; {summary['completed_new_episodes']}/288 new episodes.",
        "",
        "| Start condition | Prompt | Complete pairs | Correct pairs | CHECK macro | Repeated-action items | Forced endings |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for key, c in cells.items():
        arm, prompt = key.split("_", 1)
        macro = f"{c['check_macro']:.1%}" if c["check_macro"] is not None else "pending"
        lines.append(
            f"| {arm} | {prompt} | {c['completed_pairs']}/72 | {c['pair_correct']} | {macro} | {c['repeated_action_items']}/{c['completed_items']} | {json.dumps(c['forced_endings'])} |"
        )
    lines += [
        "",
        "CHECK macro averages C, D, L, C+D, L+D and D+door. Controls remain separate.",
        "Original-prompt cells reuse completed frozen historical runs. New cells are interleaved by pair.",
        "",
        "| Family | Start original | Start procedure | Handoff original | Handoff procedure |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in sorted({p["family"] for p in plan["pairs"].values()}):
        values = [
            cells[k]["family"][f]
            for k in (
                "start_original",
                "start_procedure",
                "handoff_original",
                "handoff_procedure",
            )
        ]
        lines.append(
            "| "
            + f.replace("|", "\\|")
            + " | "
            + " | ".join((f"{v['correct']}/{v['pairs']}" for v in values))
            + " |"
        )
    lines += [
        "",
        f"Fresh handoff-image QA: {qa_summary['pair_correct']}/{qa_summary['completed_pairs']} complete pairs.",
        "On the fixed previous 44-pair QA-correct diagnostic subset: "
        + ", ".join(
            (
                f"{k} {v['pair_correct']}/{v['completed_pairs']} complete pairs"
                for k, v in conditional.items()
            )
        )
        + ".",
        "",
        "No training, privileged readiness input, hard repeat blocking or model-score-based task selection.",
        "Prompt wording is a bundled intervention. Input length increases; token counts are in comparison.json.",
        "Actual first images and full model traces are retained for rendering and evidence audits.",
        "This is a development-set intervention. Improved handoff alone does not demonstrate autonomous acquisition.",
    ]
    tmp = out / "PROCEDURE_PROMPT_REPORT.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(out / "PROCEDURE_PROMPT_REPORT.md")
    return summary


def ask(model, processor, state, history):
    import torch
    from PIL import Image

    prompt = procedure_prompt(state, history)
    image = Image.open(io.BytesIO(base64.b64decode(state["frame_png_b64"]))).convert(
        "RGB"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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
        generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    n = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(generated[:, n:], skip_special_tokens=True)[0]
    parsed = parse_action(raw)
    name, kw = parsed or ("Answer", {"answer": "no", "confidence": 0.0})
    return {
        "action": name,
        "args": kw,
        "parse_failed": parsed is None,
        "raw_reply": raw,
        "prompt": prompt,
        "input_tokens": n,
        "output_tokens": generated.shape[1] - n,
        "frame_sha256_before": state["frame_sha256"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument(
        "--eval-manifest",
        type=Path,
        default=ROOT / "configs/epact/evaluation/full.json",
    )
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
    ap.add_argument("--baseline-dir", type=Path, default=BASE)
    ap.add_argument("--handoff-dir", type=Path, default=HANDOFF)
    ap.add_argument("--port", type=int, default=7387)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base, handoff = (args.baseline_dir, args.handoff_dir)
    plan = prepare(args.eval_manifest, args.dataset, args.data_root)
    if plan_semantics(plan) != plan_semantics(
        json.loads((handoff / "plan.json").read_text())
    ):
        raise ValueError("historical handoff plan mismatch")
    model_path = args.model_path
    sources = [
        "src/epact/diagnostics/procedure.py",
        "src/epact/evaluation/protocol.py",
        "src/epact/evaluation/policy.py",
        "src/epact/runtime/client.py",
        "src/epact/diagnostics/witness.py",
        "src/epact/diagnostics/readability.py",
        "src/epact/runtime/server.py",
        "src/epact/data/paths.py",
    ]
    old_hashes = {
        str(p.resolve()): digest(p)
        for folder in (base, handoff)
        for p in [folder / "manifest.json"]
        + [folder / "episodes" / f"{tid}.json" for tid in plan["members"]]
    }
    reference_hashes = {}
    for tid in plan["members"]:
        previous = json.loads((base / "episodes" / f"{tid}.json").read_text())
        for image_path in (
            base / "frames" / f"{previous['decided'][0]['frame_sha256_before']}.png",
            handoff / "handoff_frames" / f"{tid}.png",
        ):
            reference_hashes[str(image_path.resolve())] = digest(image_path)
    for folder in (base, handoff):
        config = json.loads((folder / "manifest.json").read_text())
        if config["model_config_sha256"] != digest(model_path / "config.json"):
            raise ValueError("historical model mismatch")
    frozen = {
        "plan": plan,
        "procedure": PROCEDURE,
        "order": decision_order(plan),
        "sources": {p: digest(ROOT / p) for p in sources},
        "historical_files": old_hashes,
        "reference_images": reference_hashes,
        "model_path": str(model_path.resolve()),
        "model_config_sha256": digest(model_path / "config.json"),
        "max_new_tokens": 96,
        "qa_max_new_tokens": 48,
        "do_sample": False,
    }
    path = out / "manifest.json"
    frozen = json.loads(json.dumps(frozen))
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("frozen experiment changed; use a new output directory")
    atomic(path, frozen)
    old = historical(plan, base, handoff)
    for arm in ARMS:
        (out / arm / "episodes").mkdir(parents=True, exist_ok=True)
    refresh_report(out, plan, old)
    print("FROZEN: 72 pairs, 288 new episodes, one fixed public procedure", flush=True)
    if args.prepare_only:
        return
    link = ServerLink(args.port)
    if args.smoke_only:
        seen = set()
        rows = []
        for pair in plan["pairs"].values():
            if pair["family"] in seen:
                continue
            seen.add(pair["family"])
            for arm in ARMS:
                for tid in pair["ids"]:
                    state, history = initial_state(link, plan["members"][tid], arm)
                    if not procedure_prompt(state, history).endswith(PROCEDURE):
                        raise ValueError(
                            "Numerical validation failed at source line 222"
                        )
                    rows.append(
                        {"task_id": tid, "arm": arm, "prefix_steps": len(history)}
                    )
                    print("SMOKE " + json.dumps(rows[-1]), flush=True)
        atomic(out / "smoke.json", rows)
        return
    runtime = {
        p: importlib.metadata.version(p) for p in ("torch", "transformers", "Pillow")
    }
    runtime_path = out / "runtime.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise ValueError("runtime changed")
    atomic(runtime_path, runtime)
    pending = [
        (arm, tid)
        for arm, tid in decision_order(plan)
        if not (out / arm / "episodes" / f"{tid}.json").exists()
    ]
    if pending:
        import torch
        from PIL import Image, ImageChops, ImageStat
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading 8B once for {len(pending)} pending episodes", flush=True)
        processor = AutoProcessor.from_pretrained(model_path)
        model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to("cuda")
            .eval()
        )
        print("Model loaded", flush=True)
        old_by = {arm: {r["task_id"]: r for r in rows} for arm, rows in old.items()}
        for arm, tid in pending:
            begin = time.monotonic()
            member = plan["members"][tid]
            state, history = initial_state(link, member, arm)
            prefix_steps = len(history)
            image = Image.open(
                io.BytesIO(base64.b64decode(state["frame_png_b64"]))
            ).convert("RGB")
            imgpath = out / arm / "initial_frames" / f"{tid}.png"
            imgpath.parent.mkdir(exist_ok=True)
            image.save(imgpath)
            before = old_by[arm][tid]
            reference = (
                handoff / "handoff_frames" / f"{tid}.png"
                if arm == "handoff"
                else base
                / "frames"
                / f"{before['decided'][0]['frame_sha256_before']}.png"
            )
            refimage = Image.open(reference).convert("RGB")
            if image.size != refimage.size:
                raise ValueError("historical input image dimensions changed")
            mae = sum(ImageStat.Stat(ImageChops.difference(image, refimage)).mean) / 3
            qa = (
                fresh_qa(model, processor, image, member["query"])
                if arm == "handoff"
                else None
            )
            start_record = {
                "image_sha256": digest(imgpath),
                "reference_image_sha256": digest(reference),
                "reference_path": str(reference),
                "reference_mae": mae,
                "frame_sha256": state["frame_sha256"],
                "history": history.copy(),
            }
            decided = []
            outcome = None
            forced = None
            for _ in range(MAX_STEPS - prefix_steps):
                step = ask(model, processor, state, history)
                decided.append(step)
                step["geometric_before"] = state.get("privileged_determinable")
                if step["parse_failed"]:
                    forced = "parse_failure"
                reply = link.call(cmd="act", action=step["action"], args=step["args"])
                if reply["terminated"]:
                    outcome = reply["outcome"]
                    break
                step["result"] = reply["result"]
                history.append(
                    {"action": step["action"], "args": step["args"], **reply["result"]}
                )
                state = advance_state(state, reply)
            if outcome is None:
                forced = "step_limit"
                outcome = link.call(
                    cmd="act", action="Answer", args={"answer": "no", "confidence": 0.0}
                )["outcome"]
            row = {
                "task_id": tid,
                "pair_id": member["pair_id"],
                "family": member["family"],
                "arm": arm,
                "answer": member["answer"],
                "prefix_steps": prefix_steps,
                "initial": start_record,
                "fresh_qa": qa,
                "decided": decided,
                "outcome": outcome,
                "forced_answer": forced,
                **action_stats(decided),
                **finish_score(decided, outcome, forced, member["answer"]),
                "seconds": round(time.monotonic() - begin, 3),
            }
            atomic(out / arm / "episodes" / f"{tid}.json", row)
            summary = refresh_report(out, plan, old)
            print(
                f"[{summary['completed_new_episodes']}/288] {arm} {tid}: correct={row['scored_correct']} decisions={len(decided)} forced={forced}",
                flush=True,
            )
    summary = refresh_report(out, plan, old)
    if not summary["complete"]:
        raise RuntimeError("incomplete experiment")
    link.file.close()
    link.sock.close()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
