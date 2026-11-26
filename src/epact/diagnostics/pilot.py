"""Historical raw/wrapper/scaffold pilot. Current benchmark entry points live in epact.evaluation."""

from __future__ import annotations
from epact.runtime.client import ServerLink, advance_state
from epact.evaluation.protocol import PROMPT, MAX_STEPS, parse_action, policy_prompt
from epact.evaluation.policy import ask_model
import argparse
import base64
import io
import json
import re
import socket
import time
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7301)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("raw", "wrapper", "scaffold"),
        default="raw",
        help="scaffold = wrapper + client-side premature-answer blocking (teacher collection: every recorded answer is determination-gated or tagged forced)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--save-teacher-rows", action="store_true")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument(
        "--task-ids",
        type=Path,
        default=None,
        help="JSON list of task ids; --start/--limit slice this list",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.task_ids is not None:
        all_ids = json.loads(args.task_ids.read_text(encoding="utf-8"))
        task_ids = all_ids[args.start : args.start + args.limit]
    else:
        task_ids = []
        with (args.data_root / f"{args.split}.jsonl").open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < args.start:
                    continue
                task_ids.append(json.loads(line)["task_id"])
                if len(task_ids) >= args.limit:
                    break
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload().eval()
    link = ServerLink(args.port)
    outcomes = []
    teacher_rows = []
    log_path = args.output_dir / f"episodes_{args.mode}.jsonl"
    t_all = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        for task_id in task_ids:
            state = link.call(cmd="start", task_id=task_id)
            history: list[dict] = []
            taken: set[tuple] = set()
            decided: list = []
            outcome = None
            scaffolded = args.mode in ("wrapper", "scaffold")
            for _ in range(MAX_STEPS):
                det = bool(state.get("privileged_determinable"))
                forced = None
                if scaffolded and det:
                    name, act_args = ask_model(
                        model, processor, state, history, args, final=True
                    )
                    if name != "Answer":
                        name, act_args = (
                            "Answer",
                            {"answer": "yes", "confidence": 0.5},
                        )
                        forced = "final"
                else:
                    name, act_args = ask_model(
                        model, processor, state, history, args, final=False
                    )
                if args.mode == "scaffold" and name == "Answer" and (not det):
                    decided.append(
                        {
                            "action": name,
                            "args": act_args,
                            "determinable": det,
                            "forced": "premature_blocked",
                        }
                    )
                    history.append(
                        {
                            "action": name,
                            "args": act_args,
                            "ok": False,
                            "error": "refused: not determinable yet — keep exploring",
                        }
                    )
                    continue
                if scaffolded and name != "Answer":
                    key = (name, act_args.get("target"))
                    if key in taken:
                        if args.mode == "scaffold":
                            decided.append(
                                {
                                    "action": name,
                                    "args": act_args,
                                    "determinable": det,
                                    "forced": "repeat_blocked",
                                }
                            )
                            history.append(
                                {
                                    "action": name,
                                    "args": act_args,
                                    "ok": False,
                                    "error": "refused: already tried — try something else",
                                }
                            )
                            continue
                        name, act_args = ("Answer", {"answer": "no", "confidence": 0.4})
                        forced = "repeat"
                    taken.add(key)
                decided.append(
                    {
                        "action": name,
                        "args": act_args,
                        "determinable": det,
                        "forced": forced,
                    }
                )
                if args.save_teacher_rows:
                    teacher_rows.append(
                        {
                            "task_id": task_id,
                            "step": len(decided) - 1,
                            "query": state.get("query"),
                            "frame_sha256": state.get("frame_sha256"),
                            "history": history[-6:],
                            "determinable": det,
                            "forced": forced,
                            "action": {"action": name, "args": act_args},
                        }
                    )
                reply = link.call(cmd="act", action=name, args=act_args)
                if reply["terminated"]:
                    outcome = reply["outcome"]
                    break
                history.append({"action": name, "args": act_args, **reply["result"]})
                decided[-1]["ok"] = reply["result"]["ok"]
                decided[-1]["error"] = reply["result"]["error"][:80]
                state = advance_state(state, reply)
            if outcome is None:
                reply = link.call(
                    cmd="act", action="Answer", args={"answer": "no", "confidence": 0.0}
                )
                outcome = reply["outcome"]
            outcomes.append(outcome)
            log.write(
                json.dumps({**outcome, "decided": decided}, sort_keys=True) + "\n"
            )
            log.flush()
    by_family = defaultdict(list)
    lure = defaultdict(int)
    for o in outcomes:
        by_family[o["family"] if not o["trap"] else f"trap:{o['trap']}"].append(
            bool(o.get("correct"))
        )
        if o.get("lure_taken") is not None:
            lure[o["lure_taken"]] += 1
    summary = {
        "mode": args.mode,
        "model_path": str(args.model_path),
        "tasks": len(outcomes),
        "accuracy": round(
            sum((bool(o.get("correct")) for o in outcomes)) / max(len(outcomes), 1), 4
        ),
        "trap_lure_rate": round(lure[True] / max(lure[True] + lure[False], 1), 4)
        if lure
        else None,
        "family_accuracy": {
            fam: round(sum(v) / len(v), 3) for fam, v in sorted(by_family.items())
        },
        "seconds": round(time.perf_counter() - t_all, 1),
    }
    if teacher_rows:
        with (args.output_dir / "teacher_rows.jsonl").open("w", encoding="utf-8") as f:
            for row in teacher_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / f"summary_{args.mode}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
