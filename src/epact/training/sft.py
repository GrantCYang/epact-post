"""Train one LoRA arm on captured decision rows (rank 16, one epoch).

Uses BF16, SDPA, q/k/v/o LoRA, AdamW, microbatch size 1,
gradient accumulation 8, loss only on the action-JSON tokens. Saves the
adapter + a stage-flushed report; reload-verifies at the end.

Run:
    epact policy train --arm trajectory --model-path <model>         --rows <trajectory_rows.jsonl> --frames-dir data/epact/data/frames         --prompt-version row --output-dir runs/trajectory
"""

from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from collections import Counter
import importlib.metadata
from epact.training.support import (
    accumulation_window,
    adapter_identity,
    atomic,
    digest,
    freeze,
    row_key,
)
from research_common.paths import ROOT

PROMPT_V1 = 'You are an embodied agent in a house, answering: {query}\nThe image is your current view. Choose ONE next action as JSON.\nActions: GoTo(target: room or object id), Open(target), Close(target), Toggle(target), MoveAside(target), LookAt(target), Answer(answer: "yes"/"no", confidence: 0..1).\nHistory of your previous actions and results:\n{history}\nReply with JSON only.'
PROMPT_V2 = 'You are an embodied agent in a house, answering: {query}\nThe image is your current view. Choose ONE next action as JSON.\nActions: GoTo(target: room or object id), Open(target), Close(target), Toggle(target), MoveAside(target), LookAt(target), Answer(answer: "yes"/"no", confidence: 0..1).\nObject ids visible earlier appear in the history. Remaining budget: {budget}. Every action costs budget; answer as soon as you can determine it.\nHistory of your previous actions and results:\n{history}\nReply with JSON only, e.g. {{"action": "Open", "args": {{"target": "Fridge|2|1"}}}}'


def history_text(history) -> str:
    if not history:
        return "(none yet)"
    return "\n".join(
        (
            f"{i}. {h['action']}({h['args']}) → {('ok' if h.get('ok') else 'FAILED: ' + str(h.get('error', ''))[:60])}"
            for i, h in enumerate(history)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--prompt-version",
        choices=("v1", "v2", "row"),
        default="v1",
        help="v2 matches the eval prompt exactly (requires remaining_budget); row uses the byte-exact captured prompt on each row",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="extra directory searched first for frame PNGs",
    )
    args = parser.parse_args()
    if args.grad_accum < 1:
        raise SystemExit("grad-accum must be positive")
    prompt_template = PROMPT_V1 if args.prompt_version == "v1" else PROMPT_V2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "train_report.json"
    report: dict = {"arm": args.arm, "rows_file": str(args.rows)}

    def flush() -> None:
        tmp = report_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        tmp.replace(report_path)

    rows = [json.loads(l) for l in args.rows.open(encoding="utf-8")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if args.prompt_version != "row" or args.frames_dir is None:
        raise SystemExit(
            "This isolated rerun requires captured row prompts and explicit frames-dir"
        )
    for row in rows:
        fh = row.get("frame_sha256")
        frame = args.frames_dir / f"{fh}.png"
        if (
            not fh
            or not frame.is_file()
            or digest(frame) != fh
            or (not row.get("prompt"))
        ):
            raise SystemExit(f"missing/changed captured input: {row.get('task_id')}")
    config = {
        "arm": args.arm,
        "rows_file": str(args.rows.resolve()),
        "rows_sha256": digest(args.rows),
        "frames_dir": str(args.frames_dir.resolve()),
        "frame_hashes": sorted({r["frame_sha256"] for r in rows}),
        "model_path": str(args.model_path.resolve()),
        "model_config_sha256": digest(args.model_path / "config.json"),
        "seed": args.seed,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "max_rows": args.max_rows,
        "epochs": 1,
        "batch_size": 1,
        "prompt_version": args.prompt_version,
        "tail_policy": "mean over actual window; update at every window end including tail",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "optimizer": "AdamW defaults; gradient clipping max_norm=1.0",
        "runtime_versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "Pillow")
        },
        "sources": {
            p: digest(ROOT / p)
            for p in ["src/epact/training/sft.py", "src/epact/training/support.py"]
        },
    }
    freeze(args.output_dir / "training_config.json", config)
    report["training_config_sha256"] = digest(args.output_dir / "training_config.json")
    report["status"] = "training"
    frames_dirs = [
        ROOT / "data/epact/data/traces_v2/frames",
        ROOT / "data/epact/data/tasks_v2/frames",
    ]
    if args.frames_dir:
        frames_dirs.insert(0, args.frames_dir)

    def find_frame(fh):
        for d in frames_dirs:
            if (d / (fh + ".png")).exists():
                return d / (fh + ".png")
        return None

    by_task_frame: dict[str, str] = {}
    for r in rows:
        if r.get("frame_sha256"):
            by_task_frame.setdefault(r["task_id"], r["frame_sha256"])
    usable = []
    for r in rows:
        fh = r.get("frame_sha256") or by_task_frame.get(r["task_id"])
        if fh and find_frame(fh):
            usable.append((r, fh))
    report["rows_total"] = len(rows)
    report["rows_usable"] = len(usable)
    if len(usable) != len(rows) or not usable:
        raise SystemExit("Every frozen training row must be usable")
    flush()
    import torch
    from PIL import Image
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    report["trainable_params_m"] = round(
        sum((p.numel() for p in trainable)) / 1000000.0, 2
    )
    flush()
    losses = []
    applied_rows = Counter()
    applied_answers = Counter()
    pending_rows = []
    optimizer_updates = 0
    window_sizes = []
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for step, (row, fh) in enumerate(usable):
        image = Image.open(find_frame(fh)).convert("RGB")
        fields = {"query": row["query"], "history": history_text(row.get("history"))}
        if args.prompt_version == "v2":
            if "remaining_budget" not in row:
                raise SystemExit(
                    "--prompt-version v2 needs remaining_budget on every row; supply captured rows with the recorded remaining action budget"
                )
            fields["budget"] = row["remaining_budget"]
        if args.prompt_version == "row":
            if "prompt" not in row:
                raise SystemExit(
                    "--prompt-version row needs row['prompt'] on every row"
                )
            prompt = row["prompt"]
        else:
            prompt = prompt_template.format(**fields)
        target = json.dumps(row["action"], sort_keys=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        prompt_only = processor.apply_chat_template(
            messages[:1],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        labels[:, : prompt_only["input_ids"].shape[1]] = -100
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        out = model(**inputs, labels=labels)
        window_size, window_end = accumulation_window(
            step, len(usable), args.grad_accum
        )
        (out.loss / window_size).backward()
        losses.append(float(out.loss.detach()))
        pending_rows.append(row)
        if window_end:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
            window_sizes.append(window_size)
            for applied in pending_rows:
                key = row_key(applied)
                applied_rows[key] += 1
                if applied["kind"] == "answer":
                    applied_answers[key] += 1
            pending_rows.clear()
        if (step + 1) % 200 == 0:
            report["progress"] = {
                "step": step + 1,
                "of": len(usable),
                "loss_mean_last200": round(sum(losses[-200:]) / 200, 4),
                "seconds": round(time.perf_counter() - t0, 1),
            }
            flush()
            print(
                f"[{args.arm}] {step + 1}/{len(usable)} loss={report['progress']['loss_mean_last200']}",
                flush=True,
            )
    expected_rows = Counter((row_key(r) for r, _ in usable))
    expected_answers = Counter((row_key(r) for r, _ in usable if r["kind"] == "answer"))
    if (
        pending_rows
        or applied_rows != expected_rows
        or applied_answers != expected_answers
    ):
        raise RuntimeError("training exposure invariant failed")
    exposure = {
        "rows_applied": sum(applied_rows.values()),
        "answers_applied": sum(applied_answers.values()),
        "optimizer_updates": optimizer_updates,
        "window_sizes": window_sizes,
        "row_counts": dict(applied_rows),
        "answer_row_counts": dict(applied_answers),
        "answer_label_counts": dict(
            Counter(
                (
                    r["action"]["args"]["answer"]
                    for r, _ in usable
                    if r["kind"] == "answer"
                )
            )
        ),
    }
    atomic(args.output_dir / "exposure_audit.json", exposure)
    report.update(
        {
            k: exposure[k]
            for k in (
                "rows_applied",
                "answers_applied",
                "optimizer_updates",
                "answer_label_counts",
            )
        }
    )
    report["last_window_size"] = window_sizes[-1]
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    report["adapter_identity"] = adapter_identity(adapter_dir)
    report["train_seconds"] = round(time.perf_counter() - t0, 1)
    report["loss_first200"] = round(sum(losses[:200]) / max(len(losses[:200]), 1), 4)
    report["loss_last200"] = round(sum(losses[-200:]) / max(len(losses[-200:]), 1), 4)
    flush()
    del model, optimizer
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    PeftModel.from_pretrained(base, adapter_dir).eval()
    report["reload_ok"] = True
    report["status"] = "ok"
    flush()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
