"""Compare fixed legal observation views with the same frozen VLM.

No model outcome chooses tasks or views. Missing/over-budget views stay
missing; denominators and geometric-determinability subsets are explicit.
The same yes/no prompt and greedy decoding as the original pilot are used.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from research_common.paths import ROOT

PHASES = ("start", "legacy", "receptacle_close", "target_close")
PROMPT = 'You are looking at a house. Question: {query}\nAnswer from this single image only. Reply with JSON only: {{"answer": "yes"/"no", "confidence": 0..1}}'


def summarize(rows, pair_ids):
    output = {}
    for phase in PHASES:
        selected = [r for r in rows if r["phase"] == phase]
        indexed = {(r["pair_id"], r["task_id"]): r for r in selected}
        selected = list(indexed.values())
        complete, grounded, correct, gc = (0, 0, 0, 0)
        same = 0
        for pid in pair_ids:
            members = [r for r in selected if r["pair_id"] == pid]
            if len(members) != 2:
                continue
            complete += 1
            ok = all((r["pred"] == r["answer"] for r in members))
            correct += ok
            same += (
                members[0]["pred"] is not None
                and members[0]["pred"] == members[1]["pred"]
            )
            if all((r["determinable"] for r in members)):
                grounded += 1
                gc += ok
        output[phase] = {
            "requested_pairs": len(pair_ids),
            "missing_or_incomplete_pairs": len(pair_ids) - complete,
            "items": len(selected),
            "item_correct": sum((r["pred"] == r["answer"] for r in selected)),
            "parse_failures": sum((r["pred"] is None for r in selected)),
            "complete_pairs": complete,
            "pair_correct": correct,
            "pair_accuracy": correct / complete if complete else None,
            "same_prediction_pairs": same,
            "geometrically_grounded_pairs": grounded,
            "grounded_pair_correct": gc,
            "grounded_pair_accuracy": gc / grounded if grounded else None,
            "confusion": dict(
                Counter((f"gt_{r['answer']}_pred_{r['pred']}" for r in selected))
            ),
        }
    output["matched_view_contrasts"] = {}
    for phase in PHASES[2:]:
        by_key = {(r["task_id"], r["phase"]): r for r in rows}
        ids = [tid for tid, ph in by_key if ph == "legacy" and (tid, phase) in by_key]
        before = sum(
            (
                by_key[tid, "legacy"]["pred"] == by_key[tid, "legacy"]["answer"]
                for tid in ids
            )
        )
        after = sum(
            (by_key[tid, phase]["pred"] == by_key[tid, phase]["answer"] for tid in ids)
        )
        output["matched_view_contrasts"][phase] = {
            "same_items": len(ids),
            "legacy_correct": before,
            "new_correct": after,
        }
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument(
        "--root", type=Path, default=ROOT / "data/epact/artifacts/readability_v3_v1"
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        help="Separate model outputs while sharing fixed capture frames",
    )
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()
    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.root / "manifest.json").read_text())
    capture_report = json.loads((args.root / "capture_report.json").read_text())
    if (
        capture_report.get("failures")
        or capture_report.get("pair_ids") != manifest["pair_ids"]
        or capture_report.get("data_sha256") != manifest["data_sha256"]
    ):
        raise SystemExit(
            "capture batch incomplete or stale; finish capture before probing"
        )
    records = []
    for pid in manifest["pair_ids"]:
        path = args.root / "pairs" / f"{pid}.json"
        if not path.exists():
            raise SystemExit(f"capture result missing: {path}")
        item = json.loads(path.read_text())
        if item["data_sha256"] != manifest["data_sha256"]:
            raise SystemExit("stale capture data hash")
        for m in item["members"]:
            for phase, view in m["views"].items():
                if "image" in view and (not view.get("unavailable")):
                    records.append(
                        {
                            "pair_id": pid,
                            "task_id": m["task_id"],
                            "query": m["query"],
                            "answer": m["answer"],
                            "family": m["family"],
                            "phase": phase,
                            **view,
                        }
                    )
    if not records:
        raise SystemExit("no captured observations")
    config = {
        "model": str(args.model_path.resolve()),
        "prompt": PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "data_sha256": manifest["data_sha256"],
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    out = output_dir / "vlm_predictions.jsonl"
    existing = (
        [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    )
    if any((r.get("config_sha256") != key for r in existing)):
        raise SystemExit("stale probe output config; use a fresh output file")
    done = {(r["task_id"], r["phase"], r["image_sha256"]) for r in existing}
    pending = [
        r for r in records if (r["task_id"], r["phase"], r["image_sha256"]) not in done
    ]
    if pending:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"loading model; {len(pending)} observations pending", flush=True)
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = (
            AutoModelForImageTextToText.from_pretrained(
                args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to("cuda")
            .eval()
        )
        print("model loaded", flush=True)
        with out.open("a") as stream:
            for i, row in enumerate(pending):
                path = args.root / row["image"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
                    raise SystemExit(f"frame changed: {path}")
                image = Image.open(path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": PROMPT.format(query=row["query"])},
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
                    g = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                    )
                reply = processor.batch_decode(
                    g[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )[0]
                match = re.search('"answer"\\s*:\\s*"(yes|no)"', reply)
                result = {
                    **row,
                    "pred": match.group(1) if match else None,
                    "raw_reply": reply,
                    "config_sha256": key,
                }
                stream.write(json.dumps(result, sort_keys=True) + "\n")
                stream.flush()
                existing.append(result)
                print(
                    f"[{i + 1}/{len(pending)}] {row['task_id']} {row['phase']}: {result['pred']}",
                    flush=True,
                )
    report = {
        "config": config,
        "summary": summarize(existing, manifest["pair_ids"]),
        "interpretation": "Diagnostic only: VLM correctness does not certify or select valid benchmark tasks.",
    }
    (output_dir / "readability_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
