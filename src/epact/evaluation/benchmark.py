"""Resumable raw-model pilot on a fixed complete-pair manifest; no training.

Geometric determinability is logged for analysis, never fed to the policy.
This run does not certify semantic evidence sufficiency or select valid tasks.
"""

import argparse
import hashlib
import json
import time
import re
import importlib.metadata
from collections import Counter
from pathlib import Path
from epact.runtime.client import ServerLink
from epact.evaluation.policy import ask_model
from epact.runtime.client import advance_state
from epact.evaluation.protocol import PROMPT
from epact.evaluation.protocol import MAX_STEPS
from epact.training.support import adapter_identity
from epact.evaluation.behavior import behavior
from epact.data.paths import house_dataset, resolve_observations


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows, pairs):
    by_id = {r["task_id"]: r for r in rows}
    complete = [ids for ids in pairs.values() if all((i in by_id for i in ids))]
    pair_ok = sum((all((by_id[i]["scored_correct"] for i in ids)) for ids in complete))
    grounded_pair_ok = sum(
        (
            all(
                (
                    by_id[i]["scored_correct"] and by_id[i]["answered_at_geometric"]
                    for i in ids
                )
            )
            for ids in complete
        )
    )
    n = len(rows)
    return {
        "requested_pairs": len(pairs),
        "completed_pairs": len(complete),
        "missing_or_incomplete_pairs": len(pairs) - len(complete),
        "completed_items": n,
        "item_correct": sum((r["scored_correct"] for r in rows)),
        "item_accuracy": sum((r["scored_correct"] for r in rows)) / n if n else None,
        "pair_correct": pair_ok,
        "pair_accuracy": pair_ok / len(complete) if complete else None,
        "geometric_grounded_pair_correct": grounded_pair_ok,
        "geometric_grounded_pair_accuracy": grounded_pair_ok / len(complete)
        if complete
        else None,
        "no_action_items": sum((r.get("steps", 0) == 0 for r in rows)),
        "ever_observed_geometric_items": sum(
            (r["ever_observed_geometric"] for r in rows)
        ),
        "answered_at_geometric_items": sum((r["answered_at_geometric"] for r in rows)),
        "correct_at_geometric_items": sum(
            (r["scored_correct"] and r["answered_at_geometric"] for r in rows)
        ),
        "ever_evidence_proxy_items": sum(
            (r.get("ever_evidence_proxy", False) for r in rows)
        ),
        "correct_at_evidence_proxy_items": sum(
            (
                r["scored_correct"] and r.get("answered_at_evidence_proxy", False)
                for r in rows
            )
        ),
        "forced_answer_items": sum((r["forced_answer"] is not None for r in rows)),
        "status_counts": dict(Counter((r["status"] for r in rows))),
        "family": {
            f: {"items": len(v), "correct": sum((r["scored_correct"] for r in v))}
            for f in sorted({r["family"] for r in rows})
            for v in [[r for r in rows if r["family"] == f]]
        },
        "interpretation": "Pilot. Geometric flags are proxies, not semantic evidence certification.",
    }


def probe_fixed_views(model, processor, args, capture, tasks, pairs):
    import torch
    from PIL import Image
    from epact.diagnostics.readability import PROMPT as PROBE_PROMPT

    root = args.output_dir / "probe_rows"
    root.mkdir(exist_ok=True)
    rows = []
    for tid in [i for ids in pairs.values() for i in ids]:
        for phase in ("start", "witness"):
            dest = root / f"{tid}__{phase}.json"
            view = capture["observations"][tid][phase]
            if digest(view["image"]) != view["image_sha256"]:
                raise RuntimeError("certified observation changed")
            if dest.exists():
                rows.append(json.loads(dest.read_text()))
                continue
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": Image.open(view["image"]).convert("RGB"),
                        },
                        {
                            "type": "text",
                            "text": PROBE_PROMPT.format(query=tasks[tid]["query"]),
                        },
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
            reply = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
            match = re.search('"answer"\\s*:\\s*"(yes|no)"', reply)
            row = {
                "task_id": tid,
                "pair_id": tasks[tid]["pair_id"],
                "phase": phase,
                "answer": tasks[tid]["answer"],
                "pred": match.group(1) if match else None,
                "raw_reply": reply,
                "image_sha256": view["image_sha256"],
            }
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(row, indent=2) + "\n")
            tmp.replace(dest)
            rows.append(row)
            print(
                f"[fixed-view {len(rows)}/{2 * len(capture['observations'])}] {tid} {phase}: {row['pred']}",
                flush=True,
            )
    summary = {}
    for phase in ("start", "witness"):
        by_id = {r["task_id"]: r for r in rows if r["phase"] == phase}
        summary[phase] = {
            "items": len(by_id),
            "correct": sum((r["pred"] == r["answer"] for r in by_id.values())),
            "parse_failures": sum((r["pred"] is None for r in by_id.values())),
            "pairs": len(pairs),
            "pair_correct": sum(
                (
                    all((by_id[i]["pred"] == by_id[i]["answer"] for i in ids))
                    for ids in pairs.values()
                )
            ),
            "same_answer_pairs": sum(
                (
                    by_id[ids[0]]["pred"] is not None
                    and by_id[ids[0]]["pred"] == by_id[ids[1]]["pred"]
                    for ids in pairs.values()
                )
            ),
        }
    (args.output_dir / "probe_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("FIXED_VIEW_SUMMARY " + json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/balanced_existing_v4"))
    ap.add_argument(
        "--capture-root", type=Path, default=Path("artifacts/readability_v3_v1")
    )
    ap.add_argument(
        "--asset-root", type=Path, help="Root for attachment-relative image paths"
    )
    ap.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/benchmark_v3_pilot_8b")
    )
    ap.add_argument("--port", type=int, default=7313)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument(
        "--protocol", choices=["legacy-v3", "inspection-v4-dev"], default="legacy-v3"
    )
    ap.add_argument("--eval-manifest", type=Path)
    ap.add_argument(
        "--dataset",
        type=Path,
        help="ProcTHOR split file; defaults to EPACT_HOUSE_DATASET or data/external/procthor-10k/val.jsonl.gz",
    )
    ap.add_argument(
        "--hide-budget",
        action="store_true",
        help="Do not expose state-dependent remaining budget to the policy.",
    )
    ap.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="optional LoRA adapter dir; merged into the base model before eval",
    )
    args = ap.parse_args()
    data = args.data_root / "val.jsonl"
    tasks = {t["task_id"]: t for t in map(json.loads, data.read_text().splitlines())}
    capture = resolve_observations(
        json.loads(
            (args.eval_manifest or args.capture_root / "manifest.json").read_text()
        ),
        args.asset_root,
    )
    if capture["data_sha256"] != digest(data):
        raise SystemExit("data changed since capture")
    if capture.get("scene_dataset_sha256") and capture[
        "scene_dataset_sha256"
    ] != digest(house_dataset(args.dataset)):
        raise SystemExit("scene dataset changed since freeze")
    pairs = {}
    for pid in capture["pair_ids"]:
        members = [t for t in tasks.values() if t.get("pair_id") == pid]
        if len(members) != 2 or {t["answer"] for t in members} != {"yes", "no"}:
            raise SystemExit(f"incomplete pair: {pid}")
        if any(
            (
                members[0][k] != members[1][k]
                for k in ("query", "house_index", "start_pose", "split")
            )
        ):
            raise SystemExit(f"inconsistent pair: {pid}")
        pairs[pid] = [t["task_id"] for t in members]
    from research_common.paths import ROOT as source_root

    source_paths = [
        "src/epact/evaluation/benchmark.py",
        "src/epact/evaluation/protocol.py",
        "src/epact/evaluation/policy.py",
        "src/epact/runtime/client.py",
        "src/epact/runtime/server.py",
        "src/epact/environment/episode.py",
        "src/epact/environment/actions.py",
        "src/epact/environment/sim.py",
    ]
    if args.protocol == "inspection-v4-dev":
        source_paths += [
            "src/epact/environment/inspection.py",
            "src/epact/environment/evidence.py",
            "src/epact/diagnostics/readability.py",
        ]
        if not args.eval_manifest or not capture.get("visually_reviewed"):
            raise SystemExit(
                "inspection benchmark requires a visually reviewed evaluation manifest"
            )
        for p, h in capture["sources"].items():
            if digest(source_root / p) != h:
                raise SystemExit(f"environment changed since validation: {p}")
    config = {
        "protocol": args.protocol,
        "hide_budget": args.hide_budget,
        "data_sha256": digest(data),
        "model": str(args.model_path.resolve()),
        "pairs": pairs,
        "prompt": PROMPT,
        "max_steps": MAX_STEPS,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "sources": {p: digest(source_root / p) for p in source_paths},
        "eligibility": capture.get(
            "eligibility", "fixed readability pilot; no filtering by model correctness"
        ),
        "eval_manifest_sha256": digest(args.eval_manifest)
        if args.eval_manifest
        else None,
        "budget": 20.0 if args.protocol == "inspection-v4-dev" else "task-specific",
    }
    config["runtime_versions"] = {
        p: importlib.metadata.version(p) for p in ("torch", "transformers", "Pillow")
    }
    config["model_config_sha256"] = digest(args.model_path / "config.json")
    config["adapter_identity"] = adapter_identity(args.adapter_path)
    config["sources"]["src/epact/evaluation/benchmark.py"] = digest(
        source_root / "src/epact/evaluation/benchmark.py"
    )
    config["sources"]["src/epact/training/support.py"] = digest(
        source_root / "src/epact/training/support.py"
    )
    if args.adapter_path:
        training_root = args.adapter_path.parent
        config["training_provenance"] = {
            p: digest(training_root / p)
            for p in (
                "training_config.json",
                "train_report.json",
                "exposure_audit.json",
            )
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != config:
        raise SystemExit("run configuration changed; use a fresh output directory")
    manifest_path.write_text(json.dumps(config, indent=2) + "\n")
    results_dir = args.output_dir / "episodes"
    results_dir.mkdir(exist_ok=True)
    ordered = [i for ids in pairs.values() for i in ids]
    existing = {p.stem: json.loads(p.read_text()) for p in results_dir.glob("*.json")}
    if set(existing) - set(ordered):
        raise SystemExit("unexpected task in output directory")
    pending = [i for i in ordered if i not in existing]
    if pending:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading model for {len(pending)} pending episodes", flush=True)
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = (
            AutoModelForImageTextToText.from_pretrained(
                args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to("cuda")
            .eval()
        )
        if args.adapter_path:
            from peft import PeftModel

            model = (
                PeftModel.from_pretrained(model, args.adapter_path)
                .merge_and_unload()
                .eval()
            )
            print(f"LoRA adapter merged: {args.adapter_path}", flush=True)
        print("Model loaded", flush=True)
        if args.protocol == "inspection-v4-dev":
            probe_fixed_views(model, processor, args, capture, tasks, pairs)
        link = ServerLink(args.port)
        for tid in pending:
            begin = time.monotonic()
            state = link.call(cmd="start", task_id=tid)
            if state.get("protocol", "legacy-v3") != args.protocol:
                raise RuntimeError("server/client protocol mismatch")
            history, decided = ([], [])
            outcome, forced = (None, None)
            ever = bool(state["privileged_determinable"])
            answered_det = False
            ever_proxy = bool(
                (state.get("privileged_evidence_proxy") or {}).get("ready")
            )
            answered_proxy = False
            for _ in range(MAX_STEPS):
                name, kw = ask_model(
                    model, processor, state, history, args, final=False
                )
                det = bool(state["privileged_determinable"])
                if args.last_parse_failed:
                    forced = "parse_failure"
                step = {
                    "action": name,
                    "args": kw,
                    "determinable_before": det,
                    "frame_sha256_before": state["frame_sha256"],
                    "raw_reply": args.last_model_reply,
                    "parse_failed": args.last_parse_failed,
                }
                decided.append(step)
                reply = link.call(cmd="act", action=name, args=kw)
                if name == "Answer":
                    answered_det = det and forced is None
                    answered_proxy = (
                        bool(
                            (state.get("privileged_evidence_proxy") or {}).get("ready")
                        )
                        and forced is None
                    )
                if reply["terminated"]:
                    outcome = reply["outcome"]
                    break
                step["result"] = reply["result"]
                step["determinable_after"] = bool(reply["privileged_determinable"])
                ever |= step["determinable_after"]
                step["evidence_proxy_after"] = reply.get("privileged_evidence_proxy")
                ever_proxy |= bool(
                    (reply.get("privileged_evidence_proxy") or {}).get("ready")
                )
                history.append({"action": name, "args": kw, **reply["result"]})
                state = advance_state(state, reply)
            if outcome is None:
                forced = "step_limit"
                outcome = link.call(
                    cmd="act", action="Answer", args={"answer": "no", "confidence": 0.0}
                )["outcome"]
            row = {
                **outcome,
                "pair_id": tasks[tid]["pair_id"],
                "decided": decided,
                **behavior(decided),
                "forced_answer": forced,
                "scored_correct": bool(outcome.get("correct")) and forced is None,
                "ever_observed_geometric": ever,
                "answered_at_geometric": answered_det,
                "ever_evidence_proxy": ever_proxy,
                "answered_at_evidence_proxy": answered_proxy,
                "seconds": round(time.monotonic() - begin, 3),
            }
            target = results_dir / f"{tid}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, indent=2) + "\n")
            temporary.replace(target)
            existing[tid] = row
            (args.output_dir / "summary.json").write_text(
                json.dumps(summarize(list(existing.values()), pairs), indent=2) + "\n"
            )
            print(
                f"[{len(existing)}/{len(ordered)}] {tid}: correct={row['scored_correct']} steps={row['steps']} geometric={answered_det}",
                flush=True,
            )
        link.file.close()
        link.sock.close()
    rows = [existing[i] for i in ordered]
    report = summarize(rows, pairs)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "episodes_raw.jsonl").write_text(
        "".join((json.dumps(r) + "\n" for r in rows))
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
