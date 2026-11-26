"""Action/reason-first evaluation on explicit twin pairs, without training.

Prompt, parser, 14-turn limit and forced-answer scoring follow the preserved
Step-0 runner. Run identity, input validation and stable seeds are explicit.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from epact.data.io import read_rows, read_json
from epact.evaluation.protocol import (
    MAX_STEPS,
    REASON_SWAP,
    parse_action,
    parse_action_reason,
    policy_prompt,
)
from epact.evaluation.behavior import behavior, open_stats
from epact.runtime.client import ServerLink, advance_state
from epact.training.support import adapter_identity
from research_common.paths import ROOT
from research_common.run_state import atomic_json, model_identity, prepare_run, sha256


def select_tasks(rows, pair_ids):
    """Reject duplicate/missing members and mismatched twin task context."""
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Pair selection must be nonempty and unique")
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate task_id")
    tasks = {r["task_id"]: r for r in rows if r["pair_id"] in pair_ids}
    groups = defaultdict(list)
    for row in tasks.values():
        if (
            "/" in row["task_id"]
            or "\\" in row["task_id"]
            or row["task_id"] in (".", "..")
        ):
            raise ValueError("Task ID is not a filename component")
        groups[row["pair_id"]].append(row)
    if set(groups) != set(pair_ids):
        raise ValueError("Requested pair is absent from task file")
    for members in groups.values():
        if len(members) != 2 or {r["answer"] for r in members} != {"yes", "no"}:
            raise ValueError("Each pair must have one yes and one no member")
        for field in ("query", "house_index", "start_pose", "split"):
            if members[0][field] != members[1][field]:
                raise ValueError("Twin context differs: " + field)
    return tasks


def rollout_seed(task_id, rollout, seed):
    value = json.dumps([task_id, rollout, seed], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") & 2147483647


def run_episode(link, policy, task, rollout, seed, mode, temperature):
    """Execute one episode; policy and simulator can be substituted in CPU tests."""
    begin = time.monotonic()
    state = link.call(cmd="start", task_id=task["task_id"])
    if state.get("protocol") != "inspection-v4-dev":
        raise ValueError("Unexpected simulator protocol")
    history, decided = ([], [])
    outcome, forced = (None, None)
    first_seed = rollout_seed(task["task_id"], rollout, seed)
    parser = parse_action_reason if mode == "reason" else parse_action
    for index in range(MAX_STEPS):
        text = policy_prompt(state, history, False, True)
        if mode == "reason":
            text = text.replace("Reply with JSON only", REASON_SWAP)
        raw = policy(state, text, first_seed + index)
        parsed = parser(raw)
        if parsed is None:
            forced = "parse_failure"
        action, args = parsed or ("Answer", {"answer": "no", "confidence": 0.0})
        decided.append(
            {
                "action": action,
                "args": args,
                "raw_reply": raw,
                "parse_failed": parsed is None,
                "frame_sha256_before": state.get("frame_sha256"),
            }
        )
        reply = link.call(cmd="act", action=action, args=args)
        if reply["terminated"]:
            outcome = reply["outcome"]
            break
        decided[-1]["result"] = reply["result"]
        history.append({"action": action, "args": args, **reply["result"]})
        state = advance_state(state, reply)
    if outcome is None:
        forced = "step_limit"
        outcome = link.call(
            cmd="act", action="Answer", args={"answer": "no", "confidence": 0.0}
        )["outcome"]
    return {
        **outcome,
        "task_id": task["task_id"],
        "pair_id": task["pair_id"],
        "rollout": rollout,
        "decided": decided,
        **behavior(decided),
        "forced_answer": forced,
        "scored_correct": bool(outcome.get("correct")) and forced is None,
        "temperature": temperature,
        "mode": mode,
        "rollout_seed": first_seed,
        "seconds": round(time.monotonic() - begin, 3),
    }


class LocalPolicy:
    def __init__(self, model, adapter, temperature, max_new_tokens, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.device = (torch, device)
        self.temperature, self.max_new_tokens = (temperature, max_new_tokens)
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        if adapter:
            from peft import PeftModel

            self.model = (
                PeftModel.from_pretrained(self.model, adapter, local_files_only=True)
                .merge_and_unload()
                .eval()
            )

    def __call__(self, state, prompt, seed):
        import base64
        import io
        from PIL import Image

        with Image.open(io.BytesIO(base64.b64decode(state["frame_png_b64"]))) as image:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )
        return self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]


def completed(path, task, rollout):
    try:
        row = read_json(path)
        return (
            "ERROR" not in row
            and row["task_id"] == task["task_id"]
            and (row["pair_id"] == task["pair_id"])
            and (row["rollout"] == rollout)
            and isinstance(row["decided"], list)
            and isinstance(row["status"], str)
            and (type(row["scored_correct"]) is bool)
            and (
                row["scored_correct"]
                == (bool(row["correct"]) and row["forced_answer"] is None)
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def summary(rows, tasks, expected):
    paired = defaultdict(dict)
    for row in rows:
        paired[row["pair_id"], row["rollout"]][tasks[row["task_id"]]["answer"]] = row[
            "scored_correct"
        ]
    complete = [m for m in paired.values() if set(m) == {"yes", "no"}]
    return {
        "status": "complete" if len(rows) == expected else "incomplete",
        "expected_episodes": expected,
        "completed_episodes": len(rows),
        "completed_pair_rollouts": len(complete),
        "correct_pair_rollouts": sum((all(m.values()) for m in complete)),
        "scored_correct": sum((r["scored_correct"] for r in rows)),
        "queried_container_opened_episodes": sum(
            (
                open_stats(r, tasks[r["task_id"]])["queried_container_opened"]
                for r in rows
            )
        ),
        "forced": dict(
            Counter((r["forced_answer"] for r in rows if r["forced_answer"]))
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tasks", "pair-ids", "dataset", "model-path", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--port", type=int, default=7301)
    parser.add_argument("--mode", choices=("action", "reason"), default="reason")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 for frozen-target greedy evaluation; 1 for Step-0 sampling",
    )
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else 256
        if args.mode == "reason"
        else 96
    )
    if (
        args.n_rollouts < 1
        or tokens < 1
        or (not math.isfinite(args.temperature))
        or (args.temperature < 0)
    ):
        parser.error(
            "Rollout count and token limit must be positive; temperature must be finite and nonnegative"
        )
    tasks = select_tasks(read_rows(args.tasks), read_json(args.pair_ids))
    plan = [(tid, k) for tid in sorted(tasks) for k in range(args.n_rollouts)]
    expected = {f"{tid}__r{k}" for tid, k in plan}
    identity = {
        "schema": "epact-reason-run-v1",
        "tasks": sha256(args.tasks),
        "pairs": sha256(args.pair_ids),
        "dataset": sha256(args.dataset),
        "model": model_identity(args.model_path),
        "adapter": adapter_identity(args.adapter_path),
        "mode": args.mode,
        "temperature": args.temperature,
        "n_rollouts": args.n_rollouts,
        "max_new_tokens": tokens,
        "seed": args.seed,
        "device": args.device,
        "sources": {
            p.relative_to(ROOT).as_posix(): sha256(p)
            for p in sorted((ROOT / "src").rglob("*.py"))
        },
    }
    prepare_run(args.output_dir, identity)
    episodes = args.output_dir / "episodes"
    episodes.mkdir(exist_ok=True)
    if {p.stem for p in episodes.glob("*.json")} - expected:
        raise ValueError("Output contains episodes outside the requested selection")
    pending = [
        (t, k)
        for t, k in plan
        if not completed(episodes / f"{t}__r{k}.json", tasks[t], k)
    ]
    if pending:
        with ServerLink(args.port) as link:
            server = link.call(cmd="describe")
            if (
                server.get("tasks_sha256") != identity["tasks"]
                or server.get("dataset_sha256") != identity["dataset"]
            ):
                raise ValueError(
                    "Simulator inputs differ from the requested task/dataset files"
                )
            if server.get("protocol") != "inspection-v4-dev":
                raise ValueError("Reason evaluation requires inspection-v4-dev")
            policy = LocalPolicy(
                args.model_path,
                args.adapter_path,
                args.temperature,
                tokens,
                args.device,
            )
            for tid, k in pending:
                row = run_episode(
                    link, policy, tasks[tid], k, args.seed, args.mode, args.temperature
                )
                atomic_json(episodes / f"{tid}__r{k}.json", row)
                print(f"{tid} rollout={k} correct={row['scored_correct']}", flush=True)
    rows = [
        read_json(episodes / f"{t}__r{k}.json")
        for t, k in plan
        if completed(episodes / f"{t}__r{k}.json", tasks[t], k)
    ]
    report = summary(rows, tasks, len(plan))
    atomic_json(args.output_dir / "summary.json", report)
    print(json.dumps(report))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
