"""Generate EPACT tasks over a range of staged ProcTHOR houses.

Run:
    epact data generate         --dataset data/external/procthor-10k/val.jsonl.gz --split val         --start-index 0 --num-houses 20 --seed 20260908         --data-root data/tasks_v0
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from epact.environment import generator as gen_mod
from epact.environment.generator import (
    GEN_VERSION,
    generate_house,
    generate_house_Ddoor,
    generate_house_L,
)
from epact.environment.sim import EpactSim, load_house
from epact.environment.episode import Episode
from epact.environment.pairing import construct_pair, family


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-houses", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    frames_dir = args.data_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.data_root / f"{args.split}.jsonl"
    report_path = (
        args.data_root
        / f"generation_report_{args.split}_{args.start_index}_{args.num_houses}.json"
    )
    spec_path = args.data_root / "spec.json"
    generator_sha = hashlib.sha256(Path(gen_mod.__file__).read_bytes()).hexdigest()[:16]
    if not spec_path.exists():
        spec_path.write_text(
            json.dumps(
                {
                    "generator": GEN_VERSION,
                    "generator_sha256_16": generator_sha,
                    "seed": args.seed,
                    "width": args.width,
                    "height": args.height,
                    "dataset": str(args.dataset),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    sim = EpactSim(width=args.width, height=args.height)
    consecutive_restarts = 0
    stats: Counter = Counter()
    house_seconds: list[float] = []
    failures: list[dict] = []
    started_all = time.perf_counter()
    task_count = 0
    try:
        with out_path.open("a", encoding="utf-8") as out:
            for i in range(args.start_index, args.start_index + args.num_houses):
                t0 = time.perf_counter()
                try:
                    house = load_house(args.dataset, i)
                    if sim.controller is None:
                        sim.start(house)
                    else:
                        sim.reset(house)
                    tasks = generate_house(sim, args.split, i, args.seed, frames_dir)
                    for stage, fn in (
                        ("L", generate_house_L),
                        ("Ddoor", generate_house_Ddoor),
                    ):
                        try:
                            tasks += fn(
                                sim, args.split, i, args.seed, frames_dir, house
                            )
                        except Exception as exc:
                            failures.append(
                                {
                                    "house_index": i,
                                    "stage": stage,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            sim.controller.step(action="Pass")
                except Exception as exc:
                    failures.append(
                        {"house_index": i, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    alive = sim.controller is not None
                    if alive:
                        try:
                            sim.controller.step(action="Pass")
                        except Exception:
                            alive = False
                    if not alive:
                        consecutive_restarts += 1
                        if consecutive_restarts > 5:
                            raise RuntimeError(
                                "simulator died 5 times in a row; aborting"
                            ) from exc
                        try:
                            sim.stop()
                        except Exception:
                            pass
                        sim = EpactSim(width=args.width, height=args.height)
                        print(f"[gen] restarted simulator after house {i}", flush=True)
                    continue
                consecutive_restarts = 0
                raw_tasks = tasks
                tasks = []
                construction = []
                cache = {(args.split, i): house}
                for seed_task in raw_tasks:
                    if seed_task.get("trap") or (
                        seed_task["query_type"] != "STATE"
                        and seed_task["answer"] != "yes"
                    ):
                        continue
                    try:
                        ep = Episode(sim, seed_task, args.dataset)
                        ep.setup(cache)
                        paired, record = construct_pair(seed_task, sim.objects())
                        tasks.extend(paired)
                    except Exception as exc:
                        record = {
                            "status": "unresolved",
                            "reason": type(exc).__name__ + ": " + str(exc)[:300],
                        }
                    construction.append(
                        {
                            "task_id": seed_task["task_id"],
                            "family": family(seed_task),
                            **record,
                        }
                    )
                construction_dir = args.data_root / "construction"
                construction_dir.mkdir(exist_ok=True)
                (construction_dir / f"{args.split}_{i:05d}.json").write_text(
                    json.dumps(
                        {
                            "raw_family_counts": dict(
                                Counter((family(t) for t in raw_tasks))
                            ),
                            "records": construction,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                for task in tasks:
                    out.write(json.dumps(task, sort_keys=True) + "\n")
                    key = "+".join(task["hiding_kinds"])
                    trap = task.get("trap")
                    stats[f"trap:{trap}" if trap else f"kind:{key}"] += 1
                    stats[f"query:{task['query_type']}"] += 1
                    stats[f"answer:{task.get('answer')}"] += 1
                out.flush()
                task_count += len(tasks)
                house_seconds.append(round(time.perf_counter() - t0, 2))
                print(
                    f"[gen] {args.split} house {i}: {len(tasks)} tasks in {house_seconds[-1]}s (total {task_count})",
                    flush=True,
                )
    finally:
        sim.stop()
        elapsed = time.perf_counter() - started_all
        report = {
            "generator": GEN_VERSION,
            "generator_sha256_16": generator_sha,
            "split": args.split,
            "start_index": args.start_index,
            "num_houses": args.num_houses,
            "seed": args.seed,
            "tasks_written": task_count,
            "stats": dict(sorted(stats.items())),
            "failures": failures,
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_house_mean": round(
                sum(house_seconds) / max(len(house_seconds), 1), 2
            ),
            "tasks_per_hour": round(3600 * task_count / max(elapsed, 1)),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
