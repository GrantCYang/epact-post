"""Resolve saved evaluation recipes; print commands without launching rollouts."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
from research_common.paths import ROOT


def recipes():
    return json.loads(
        (ROOT / "configs/epact/reason_runs.json").read_text(encoding="utf-8")
    )["recipes"]


def plan(name, archive_root, dataset, model, output, port=7301, check_inputs=False):
    recipe = recipes()[name]
    archive_root = Path(archive_root).resolve()
    tasks = archive_root / recipe["tasks"]
    pair_ids = ROOT / recipe["pair_ids"]
    if hashlib.sha256(pair_ids.read_bytes()).hexdigest() != recipe["pair_ids_sha256"]:
        raise ValueError("Bundled pair selection identity differs")
    pairs = json.loads(pair_ids.read_text())
    if len(pairs) != recipe["pairs"] or len(set(pairs)) != len(pairs):
        raise ValueError("Pair count or identity differs")
    adapter = archive_root / recipe["adapter"] if recipe["adapter"] else None
    if check_inputs:
        if hashlib.sha256(tasks.read_bytes()).hexdigest() != recipe["tasks_sha256"]:
            raise ValueError("Restored task identity differs")
        from epact.evaluation.reason import select_tasks

        rows = [
            json.loads(line) for line in tasks.read_text().splitlines() if line.strip()
        ]
        select_tasks(rows, pairs)
        if (
            not Path(dataset).is_file()
            or not Path(model).is_dir()
            or (adapter and (not adapter.is_dir()))
        ):
            raise ValueError("Dataset, model or recipe adapter is absent")
    server = [
        "epact",
        "policy",
        "serve",
        "--data-root",
        str(tasks.parent),
        "--split",
        recipe["split"],
        "--dataset",
        str(dataset),
        "--protocol",
        "inspection-v4-dev",
        "--port",
        str(port),
    ]
    evaluate = [
        "epact",
        "policy",
        "evaluate",
        "--tasks",
        str(tasks),
        "--pair-ids",
        str(pair_ids),
        "--dataset",
        str(dataset),
        "--model-path",
        str(model),
        "--output-dir",
        str(output),
        "--port",
        str(port),
    ]
    for key in ("mode", "temperature", "n_rollouts", "max_new_tokens", "seed"):
        evaluate += ["--" + key.replace("_", "-"), str(recipe[key])]
    if adapter:
        evaluate += ["--adapter-path", str(adapter)]
    return {
        "recipe": name,
        "pairs": recipe["pairs"],
        "tasks_sha256": recipe["tasks_sha256"],
        "inputs_checked": check_inputs,
        "executed": False,
        "server_argv": server,
        "evaluate_argv": evaluate,
        "server_posix": shlex.join(server),
        "evaluate_posix": shlex.join(evaluate),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=recipes())
    for name in ("archive-root", "dataset", "model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--port", type=int, default=7301)
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Check restored tasks, selected twins and required paths, without loading models",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            plan(
                args.recipe,
                args.archive_root,
                args.dataset,
                args.model,
                args.out,
                args.port,
                args.check_inputs,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
