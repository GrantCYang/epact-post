"""Browse the paper's studies and recompute their recorded comparisons on CPU."""

import argparse
import json
from pathlib import Path
from research_common.paths import ROOT
from research_common.results import recompute, verify_report

POLICIES = {
    "base": "Frozen instruct model",
    "a0": "No-Open SFT, seed 1",
    "a1": "Source-family RL from A0",
    "a2": "Matched-step source SFT from A0",
    "p": "Container-family RL from A0",
    "s2": "No-Open SFT, seed 2",
    "s3": "No-Open SFT, seed 3",
    "ans": "Answer-row-only diagnostic",
}
POST = {
    "told_profile": "Stated profile, immediate",
    "told_profile_reasoned": "Stated profile, reasoned",
    "checkup_reasoned": "Autonomous checkup, K12",
    "checkup_reasoned_k40": "Autonomous checkup, K40",
    "checkup_watch_reasoned": "Watched checkup, reasoned",
}


def catalog():
    return json.loads((ROOT / "configs/studies.json").read_text(encoding="utf-8"))[
        "studies"
    ]


def replay(study):
    result = recompute()
    checked = verify_report(result)
    result["verified_report_values"] = checked
    result["new_model_inference"] = False
    if study == "epact":
        result.pop("post")
    elif study == "post":
        result = {
            k: result[k]
            for k in (
                "input_sha256",
                "post",
                "verified_report_values",
                "new_model_inference",
            )
        }
    return result


def display(result):
    lines = ["Recorded-result replay; no model inference.", ""]
    if "target_combined" in result:
        lines += [
            "EPACT: frozen 55-pair reason-first target evaluation",
            "| Policy | Correct pairs | Queried-container openings |",
            "| --- | ---: | ---: |",
        ]
        for key, label in POLICIES.items():
            row = result["target_combined"][key]
            lines.append(
                f"| {label} ({key}) | {row['pair_success']}/{row['pairs']} | {row['queried_container_opened_eps']}/{row['episodes']} |"
            )
        lines.append("")
    if "post" in result:
        lines += [
            "POST: paired body-capability choices",
            "| Condition | Correct pairs | Correct choices |",
            "| --- | ---: | ---: |",
        ]
        for key, label in POST.items():
            row = result["post"][key]
            lines.append(
                f"| {label} | {row['pair_correct']}/{row['pairs']} | {row['correct_choices']}/{row['episodes']} |"
            )
    lines += [
        "",
        f"Verified {result['verified_report_values']} frozen report values across the combined evidence bundle.",
        "See data/README.md for denominators, protocol differences and limitations.",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "list", help="List research blocks and their reproduction entry points"
    )
    show = commands.add_parser(
        "show", help="Show inputs, conditions, outputs and execution coverage"
    )
    show.add_argument("study", choices=catalog())
    run = commands.add_parser(
        "replay", help="Recompute the included EPACT/POST scoring records"
    )
    run.add_argument("--study", choices=("all", "epact", "post"), default="all")
    run.add_argument("--out", type=Path)
    run.add_argument(
        "--json", action="store_true", help="Print the complete machine-readable result"
    )
    args = parser.parse_args(argv)
    if args.command == "list":
        for key, study in catalog().items():
            print(f"{key}: {study['title']}\n  {study['replay_command']}")
    elif args.command == "show":
        print(json.dumps(catalog()[args.study], indent=2))
    else:
        result = replay(args.study)
        if args.out:
            if args.out.exists():
                raise ValueError("Use a new output path")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2) if args.json else display(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
