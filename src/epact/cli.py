"""EPACT: data, adaptation, evaluation and failure-analysis workflows."""

import argparse
import json
import sys
from pathlib import Path
from research_common.paths import ROOT
from research_common.cli import run_module
from epact.data.catalog import ExperimentCatalog
from epact.data.validate import data_check
from epact.evaluation.original import results


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: epact {original,policy,data,diagnose} [command] [options]")
        print(
            "  original  Original action-only benchmark presets: data-check/results/train/serve/evaluate"
        )
        print("  policy    Explicit-file serve/evaluate/train, plus frozen recipe plan")
        print("  data      Task construction, review and demonstrations")
        print("  diagnose  Evidence-use and acquisition failure analysis")
        print("Read recorded EPACT and POST comparisons with: evidence replay")
        print("Older top-level commands remain compatibility aliases; see README.md.")
        return 0
    if argv and argv[0] == "original":
        if len(argv) == 1 or argv[1] in ("-h", "--help"):
            print(
                "Usage: epact original {data-check,results,train,serve,evaluate} [options]"
            )
            return 0
        if argv[1] not in ("data-check", "results", "train", "serve", "evaluate"):
            raise SystemExit("Unknown original benchmark command: " + argv[1])
        return main(["server" if argv[1] == "serve" else argv[1], *argv[2:]])
    if argv and argv[0] == "policy":
        commands = {
            "serve": "epact.runtime.server",
            "evaluate": "epact.evaluation.reason",
            "train": "epact.training.sft",
            "plan": "epact.evaluation.recipe",
        }
        if len(argv) == 1 or argv[1] in ("-h", "--help"):
            print("Usage: epact policy {serve,evaluate,train,plan} [options]")
            print(
                "Explicit-file protocols; plan resolves a frozen recipe without execution."
            )
            return 0
        if argv[1] not in commands:
            raise SystemExit("Unknown policy command: " + argv[1])
        return run_module(commands[argv[1]], argv[2:])
    direct = {
        "evaluate-reason": "epact.evaluation.reason",
        "serve": "epact.runtime.server",
        "train-rows": "epact.training.sft",
    }
    groups = {
        "data": {
            "frozen": "frozen",
            "generate": "tasks",
            "pool": "pool",
            "audit": "audit",
            "witnesses": "witnesses",
            "select": "selection",
            "demonstrations": "demonstrations",
        },
        "diagnose": {
            "pilot": "pilot",
            "readability": "readability",
            "witness": "witness",
            "procedure": "procedure",
            "report": "report",
        },
    }
    if argv and argv[0] in direct:
        return run_module(direct[argv[0]], argv[1:])
    if argv and argv[0] in groups:
        group = groups[argv[0]]
        if len(argv) == 1 or argv[1] in ("-h", "--help"):
            print("Usage: epact " + argv[0] + " {" + ",".join(group) + "} [options]")
            return 0
        if argv[1] not in group:
            raise SystemExit("Unknown " + argv[0] + " command: " + argv[1])
        package = "data" if argv[0] == "data" else "diagnostics"
        return run_module("epact." + package + "." + group[argv[1]], argv[2:])
    catalog = ExperimentCatalog.load()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "data-check", help="Validate bundled datasets without a simulator/model"
    )
    check.add_argument("--verify-images", action="store_true")
    check.add_argument("--data-root", type=Path, help="Restored EPACT attachment root")
    check.add_argument("--output", type=Path)
    summary = sub.add_parser(
        "results",
        help="Compatibility alias: original action-only results; use evidence replay for the combined paper",
    )
    summary.add_argument("--output", type=Path)
    summary.add_argument(
        "--data-root", type=Path, help="Restored EPACT attachment root"
    )
    train = sub.add_parser(
        "train", help="Train one LoRA arm from the supplied base model"
    )
    train.add_argument("--experiment", choices=catalog.conditions, default="full")
    train.add_argument("--arm", choices=["trajectory", "answer"], default="trajectory")
    train.add_argument(
        "--seed", type=int, choices=catalog.seeds, default=catalog.seeds[0]
    )
    train.add_argument("--model-path", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--data-root", type=Path, help="Restored EPACT attachment root")
    for name in ["evaluate", "server"]:
        command = sub.add_parser(name)
        command.add_argument("--experiment", choices=catalog.conditions, default="full")
        command.add_argument("--port", type=int, default=7301)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument(
            "--data-root", type=Path, help="Restored EPACT attachment root"
        )
        if name == "evaluate":
            command.add_argument("--model-path", type=Path, required=True)
            command.add_argument("--adapter-path", type=Path)
            command.add_argument("--output-dir", type=Path, required=True)
    for name, help in [
        (
            "evaluate-reason",
            "Evaluate the saved reason-first/action protocol on explicit task pairs",
        ),
        ("serve", "Serve an explicit task JSONL split"),
        ("train-rows", "Train from explicit demonstration rows"),
        ("data", "Task construction, witness selection and demonstration preparation"),
        ("diagnose", "Failure analysis and controlled evidence probes"),
    ]:
        sub.add_parser(name, help=help)
    sub.add_parser(
        "original",
        help="Original action-only benchmark presets: data-check/results/train/serve/evaluate",
    )
    sub.add_parser(
        "policy", help="Explicit-file policy execution and frozen recipe planning"
    )
    args = parser.parse_args(argv)
    catalog = ExperimentCatalog.load(data_root=args.data_root)
    if args.command in ("data-check", "results"):
        report = (
            data_check(args.verify_images, catalog)
            if args.command == "data-check"
            else results(catalog)
        )
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0
    if args.command == "train":
        rows = catalog.experiment(args.experiment) / "data" / (args.arm + "_rows.jsonl")
        if not rows.exists():
            parser.error(f"No bundled {args.arm} arm for {args.experiment}")
        return run_module(
            "epact.training.sft",
            [
                "--arm",
                f"{args.experiment}_{args.arm}_s{args.seed}",
                "--rows",
                rows,
                "--frames-dir",
                catalog.data_root / "data/frames",
                "--seed",
                args.seed,
                "--model-path",
                args.model_path.resolve(),
                "--output-dir",
                args.output_dir.resolve(),
                "--prompt-version",
                "row",
            ],
        )
    eroot = catalog.eval_root(args.experiment)
    common = [
        "--data-root",
        eroot / "data",
        "--port",
        args.port,
        "--protocol",
        "inspection-v4-dev",
    ]
    if args.command == "server":
        return run_module(
            "epact.runtime.server", common + ["--dataset", args.dataset.resolve()]
        )
    command = common + [
        "--eval-manifest",
        ROOT / "configs/epact/evaluation" / (args.experiment + ".json"),
        "--capture-root",
        catalog.frozen,
        "--asset-root",
        catalog.data_root,
        "--model-path",
        args.model_path.resolve(),
        "--output-dir",
        args.output_dir.resolve(),
        "--dataset",
        args.dataset.resolve(),
        "--hide-budget",
    ]
    if args.adapter_path:
        command += ["--adapter-path", args.adapter_path.resolve()]
    return run_module("epact.evaluation.benchmark", command)


if __name__ == "__main__":
    raise SystemExit(main())
