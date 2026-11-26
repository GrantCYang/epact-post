"""Original 72-pair benchmark and SFT result reconstruction."""

from collections import Counter, defaultdict
from pathlib import Path
from epact.data.catalog import ExperimentCatalog
from epact.data.io import read_json, read_rows


def summarize_episodes(directory, tasks):
    """Pair correctness and executed-action counts from raw episode records."""
    episodes = {p.stem: read_json(p) for p in Path(directory).glob("*.json")}
    groups = defaultdict(list)
    for task in tasks:
        groups[task["pair_id"]].append(task)
    family = defaultdict(Counter)
    for members in groups.values():
        stats = family[members[0]["benchmark_family"]]
        stats["expected_pairs"] += 1
        found = [
            (episodes[t["task_id"]], t) for t in members if t["task_id"] in episodes
        ]
        if len(found) == 2:
            stats["completed_pairs"] += 1
            stats["correct_pairs"] += int(all((e["scored_correct"] for e, _ in found)))
        for ep, task in found:
            decisions = ep["decided"]
            stats["episodes"] += 1
            stats["no_action"] += int(
                not any((d["action"] != "Answer" for d in decisions))
            )
            stats["forced"] += int(bool(ep.get("forced_answer")))
            for action in ("Open", "Toggle"):
                stats[f"{action.lower()}_attempt_episodes"] += int(
                    any((d["action"] == action for d in decisions))
                )
                if action == "Open":
                    successful = any(
                        (
                            s["action"] == "Open"
                            and s.get("ok") is True
                            and (
                                s.get("args", {}).get("target")
                                == task.get("receptacle_id")
                            )
                            and any(
                                (
                                    n.get("action") == "OpenObject"
                                    and n.get("success") is True
                                    and (n.get("objectId") == task.get("receptacle_id"))
                                    for n in s.get("native_steps", [])
                                )
                            )
                            for s in ep.get("step_log", [])
                        )
                    )
                else:
                    successful = any(
                        (
                            s["action"] == "Toggle" and s.get("ok") is True
                            for s in ep.get("step_log", [])
                        )
                    )
                stats[f"{action.lower()}_success_episodes"] += int(successful)
    return {name: dict(stats) for name, stats in sorted(family.items())}


def results(catalog=None):
    catalog = catalog or ExperimentCatalog.load()
    report = {}
    for name in catalog.conditions:
        eroot = catalog.eval_root(name)
        selected = set(read_json(eroot / "manifest.json")["pair_ids"])
        tasks = [
            t for t in read_rows(eroot / "data/val.jsonl") if t["pair_id"] in selected
        ]
        arms = {
            "frozen_8b": catalog.data_root
            / "artifacts/benchmark_balanced_v4_v2/8b/episodes"
        }
        arms.update(
            {
                p.parent.name: p
                for p in (catalog.experiment(name) / "results").glob("eval_*/episodes")
            }
        )
        report[name] = {
            arm: summarize_episodes(path, tasks) for arm, path in arms.items()
        }
    return report
