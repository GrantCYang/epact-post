"""Construct a house-disjoint paired training pool from train-split seeds.

Twins share a split. Each pair remains review_pending until capture, semantic
review and oracle replay pass. Outputs are atomic and resumable; the dataset
hash and construction settings are recorded with the pool.

Run:
    epact data pool --source data/epact/data/tasks_v3/train.jsonl         --dataset data/external/procthor-10k/train.jsonl.gz --out runs/training_pool
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from research_common.paths import ROOT
from epact.environment.pairing import construct_pair, family
from epact.environment.inspection import InspectionEpisode, InspectionSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", type=Path, default=ROOT / "data/epact/data/tasks_v3/train.jsonl"
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/external/procthor-10k/train.jsonl.gz",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-family", type=int, default=25)
    ap.add_argument(
        "--families",
        nargs="+",
        default=["C", "D", "L", "C+D", "L+D", "D+door", "N", "STATE|D"],
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "scene_dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "split": "train",
        "per_family": args.per_family,
        "families": args.families,
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "src/epact/environment/pairing.py",
                ROOT / "src/epact/environment/inspection.py",
                ROOT / "src/epact/environment/episode.py",
            ]
        },
        "purpose": "SFT training pool candidates; house-disjoint from val-split evaluation pairs; construction is not validity",
    }
    spec = args.out / "spec.json"
    if spec.exists() and json.loads(spec.read_text()) != config:
        raise SystemExit("Use a fresh output directory after changes")
    spec.write_text(json.dumps(config, indent=2))
    source = [json.loads(l) for l in args.source.read_text().splitlines()]
    if any((not t["task_id"].startswith("train") for t in source)):
        raise SystemExit("non-train task id in source; split hygiene violated")
    candidates = defaultdict(list)
    for t in source:
        if t["task_id"].endswith("-twin") or t.get("trap"):
            continue
        if t["query_type"] != "STATE" and t["answer"] != "yes":
            continue
        candidates[family(t)].append(t)
    for ts in candidates.values():
        ts.sort(key=lambda t: (t["house_index"], t["task_id"]))
    records_path = args.out / "construction.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    data_path = args.out / "train.jsonl"
    pairs = (
        [json.loads(l) for l in data_path.read_text().splitlines()]
        if data_path.exists()
        else []
    )
    done = {r["task_id"] for r in records}
    counts = Counter(
        (t["benchmark_family"] for t in pairs if not t["task_id"].endswith("-twin"))
    )
    houses = defaultdict(set)
    for t in pairs:
        houses[t["benchmark_family"]].add(t["house_index"])
    sim = InspectionSim()
    cache = {}
    try:
        for f in args.families:
            for task in candidates[f]:
                if counts[f] >= args.per_family:
                    break
                if task["task_id"] in done or task["house_index"] in houses[f]:
                    continue
                try:
                    ep = InspectionEpisode(sim, task, args.dataset)
                    ep.setup(cache)
                    pair, record = construct_pair(task, sim.objects())
                except Exception as exc:
                    pair = []
                    record = {
                        "status": "unresolved",
                        "reason": type(exc).__name__ + ": " + str(exc)[:300],
                    }
                records.append({"task_id": task["task_id"], "family": f, **record})
                if pair:
                    pairs.extend(pair)
                    counts[f] += 1
                    houses[f].add(task["house_index"])
                tmp = data_path.with_suffix(".tmp")
                tmp.write_text("".join((json.dumps(t) + "\n" for t in pairs)))
                tmp.replace(data_path)
                records_path.write_text(json.dumps(records, indent=2))
                print(
                    json.dumps({"family": f, "task": task["task_id"], **record}),
                    flush=True,
                )
    finally:
        sim.stop()
    report = {
        "source_seed_counts": {f: len(candidates[f]) for f in args.families},
        "constructed_pairs": dict(counts),
        "missing_families": [f for f in args.families if counts[f] < args.per_family],
        "statuses": dict(Counter((r["status"] for r in records))),
        "note": "Construction is not task validity; every pair remains review_pending. Training pool only — never an evaluation set.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
