"""Render current handoff results; optionally watch to completion and archive."""

import argparse
import json
import time
import zipfile
from pathlib import Path
from epact.diagnostics.witness import ROOT, atomic, digest, summarize

LABELS = {
    "baseline_correct": "Unassisted baseline",
    "old_witness_correct": "Previously reviewed witness QA",
    "fresh_qa_correct": "Fresh handoff-image QA",
    "first_answer_correct": "Immediate interactive answer",
    "scored_correct": "Final assisted interaction",
}


def fraction(score):
    acc = score["accuracy"]
    return (
        f"{score['correct']}/{score['pairs']} ({acc:.1%})"
        if acc is not None
        else "pending"
    )


def report(out):
    plan = json.loads((out / "plan.json").read_text())
    rows = [json.loads(p.read_text()) for p in (out / "episodes").glob("*.json")]
    summary = summarize(rows, plan["pairs"])
    complete = summary["missing_or_incomplete_pairs"] == 0
    lines = [
        "# EPACT witness handoff diagnostic",
        "",
        f"Status: {('all episodes complete' if complete else 'partial; do not interpret as final')}. {summary['completed_items']}/{2 * len(plan['pairs'])} items; {summary['completed_pairs']}/{len(plan['pairs'])} complete pairs.",
        "",
        "The model receives an assisted, reviewed acquisition prefix and then ordinary interactive control. This is an assisted diagnostic, not an autonomous benchmark score.",
        "",
        "| Condition | Both-members-correct pairs | Equal-family CHECK macro |",
        "|---|---:|---:|",
    ]
    for key, label in LABELS.items():
        macro = summary["check_macro"][key]
        lines.append(
            f"| {label} | {fraction(summary['pair_scores'][key])} | "
            + (f"{macro:.1%}" if macro is not None else "pending")
            + " |"
        )
    lines += [
        "",
        "N and STATE controls are excluded from CHECK macro. Partial-run baseline and witness QA columns use only the same completed pairs.",
        "",
        "| Family | Pairs | Baseline | Fresh QA | First answer | Assisted final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam, stats in summary["family"].items():
        lines.append(
            f"| {fam} | {stats['scored_correct']['pairs']} | "
            + " | ".join(
                (
                    fraction(stats[k])
                    for k in (
                        "baseline_correct",
                        "fresh_qa_correct",
                        "first_answer_correct",
                        "scored_correct",
                    )
                )
            )
            + " |"
        )
    n = len(rows)
    lines += [
        "",
        f"Immediate Answer actions: {summary['first_answer_items']}/{n} items. Items continuing to act: {summary['continued_items']}/{n}. Items repeating an identical model action: {summary['repeated_action_items']}/{n}.",
        f"Forced endings (never credited): {json.dumps(summary['forced_endings'], sort_keys=True)}.",
        "",
        "On the post-hoc subset where fresh-image QA answers both twins correctly:",
    ]
    for key in ("first_answer_correct", "scored_correct"):
        lines.append(
            f"- {LABELS[key]}: {fraction(summary['fresh_qa_correct_pair_subset'][key])}."
        )
    changed = sorted(
        (r for r in rows if r["witness_image_mae"] > 0),
        key=lambda r: -r["witness_image_mae"],
    )
    lines += [
        "",
        f"Fresh images differ from saved reviewed witnesses in {len(changed)}/{n} items. Maximum mean absolute RGB difference (0-255 scale): {summary['image_mae_max']}.",
        "Image differences are not a leakage verdict. Fresh QA and policy see the exact same handoff image. Any changed witness still requires visual interpretation before claiming reviewed evidence sufficiency for that frame.",
        "",
        "The QA-to-policy contrast changes instructions, output schema and text history together. It cannot alone prove an isolated stopping mechanism or latent autonomous acquisition ability. The next minimal intervention compares a fixed public prompt, answer-only SFT and full-trajectory SFT from the original starts.",
        "",
        "See `plan.json`, `manifest.json` and individual `episodes/*.json` for provenance.",
    ]
    path = out / "WITNESS_HANDOFF_REPORT.md"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    atomic(
        out / "report_audit.json",
        {
            "completed": complete,
            "completed_items": n,
            "changed_images": [
                {k: r[k] for k in ("task_id", "family", "witness_image_mae")}
                for r in changed
            ],
        },
    )
    return (complete, n)


def archive(out):
    dest = out / "witness_handoff_results.zip"
    temp = dest.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in out.iterdir():
            if path.is_file() and (
                path.suffix in (".json", ".md") or path.name in ("chain.log",)
            ):
                z.write(path, path.relative_to(out))
        for directory in ("episodes", "handoff_frames"):
            for path in sorted((out / directory).glob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(out))
        for rel in (
            "src/epact/diagnostics/witness.py",
            "src/epact/diagnostics/report.py",
            "src/epact/evaluation/protocol.py",
            "src/epact/evaluation/policy.py",
            "src/epact/runtime/client.py",
            "src/epact/diagnostics/readability.py",
            "src/epact/data/paths.py",
        ):
            z.write(ROOT / rel, rel)
    temp.replace(dest)
    atomic(
        out / "archive.json",
        {"path": str(dest), "sha256": digest(dest), "bytes": dest.stat().st_size},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/epact/artifacts/witness_handoff_v1_8b",
    )
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    last_count = -1
    while True:
        complete, n = report(args.output_dir)
        if n != last_count:
            print(f"Reported {n} items; complete={complete}", flush=True)
            last_count = n
        if complete:
            archive(args.output_dir)
            print("Final report and result archive written", flush=True)
            return
        status_path = args.output_dir / "status.json"
        if (
            status_path.exists()
            and json.loads(status_path.read_text())["stage"] == "failed"
        ):
            print("Runner failed; partial report retained", flush=True)
            raise SystemExit(1)
        if not args.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
