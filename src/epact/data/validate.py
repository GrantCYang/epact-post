"""Validate restored twins, demonstrations, controls and image identities."""

from collections import Counter, defaultdict
from epact.data.catalog import ExperimentCatalog
from epact.data.io import read_json, read_rows


def data_check(verify_images=False, catalog=None):
    """Check real bundled rows, twin membership, controls, and image references."""
    catalog = catalog or ExperimentCatalog.load()
    report = {}
    images = set()
    for name, condition in catalog.conditions.items():
        npairs, nrows, nanswers = (condition[k] for k in ("pairs", "rows", "answers"))
        exp = catalog.experiment(name)
        rows = read_rows(exp / "data/trajectory_rows.jsonl")
        answers = [r for r in rows if r["kind"] == "answer"]
        pairs = defaultdict(set)
        labels = defaultdict(set)
        for row in rows:
            pairs[row["pair_id"]].add(row["task_id"])
            if row["kind"] == "answer":
                labels[row["pair_id"]].add(row["action"]["args"]["answer"])
            frame = catalog.data_root / "data/frames" / (row["frame_sha256"] + ".png")
            if not frame.is_file():
                raise ValueError(f"Missing training frame: {frame}")
            images.add(frame)
        if (len(pairs), len(rows), len(answers)) != (npairs, nrows, nanswers):
            raise ValueError(f"Unexpected {name} data counts")
        if not all(
            (len(tids) == 2 and labels[p] == {"yes", "no"} for p, tids in pairs.items())
        ):
            raise ValueError(f"Incomplete/opposite-label twin pair: {name}")
        answer_path = exp / "data/answer_rows.jsonl"
        if answer_path.exists() and read_rows(answer_path) != answers:
            raise ValueError(f"Answer exposure differs: {name}")
        actions = Counter((r["action"]["action"] for r in rows))
        families = Counter({})
        for pid in pairs:
            families[next((r["family"] for r in rows if r["pair_id"] == pid))] += 1
        if name == "no-open":
            if set(families) & {"C", "C+D", "D+door"}:
                raise ValueError("Open-bearing training family in no-open condition")
            for r in rows:
                if any((a["action"] == "Open" for a in [r["action"], *r["history"]])):
                    raise ValueError("Open supervision in no-open condition")
                history = (
                    r["prompt"]
                    .split("History of your previous actions and results:\n", 1)[1]
                    .split("\nReply with JSON only", 1)[0]
                )
                if "Open(" in history:
                    raise ValueError("Open in actual no-open prompt history")
                if "Open(target)" not in r["prompt"]:
                    raise ValueError("Public Open API missing")
        if name == "cd-light" and (
            set(families) - {"C", "D", "C+D"} or actions["Toggle"]
        ):
            raise ValueError("Invalid light-transfer training condition")
        report[name] = dict(
            pairs=len(pairs),
            rows=len(rows),
            answer_rows=len(answers),
            actions=dict(actions),
            families=dict(families),
        )
    manifest = read_json(catalog.frozen / "manifest.json")
    tasks = read_rows(catalog.frozen / "data/val.jsonl")
    selected = set(manifest["pair_ids"])
    groups = defaultdict(list)
    for task in tasks:
        if task["pair_id"] in selected:
            groups[task["pair_id"]].append(task)
    if len(groups) != 72 or not all(
        (
            len(ts) == 2 and {t["answer"] for t in ts} == {"yes", "no"}
            for ts in groups.values()
        )
    ):
        raise ValueError(
            "Frozen evaluation must contain 72 complete balanced twin pairs"
        )
    for obs in manifest["observations"].values():
        for view in obs.values():
            path = catalog.data_root / view["image"]
            if not path.is_file():
                raise ValueError(f"Missing fixed QA image: {path}")
            images.add(path)
    if verify_images:
        from PIL import Image

        for path in images:
            with Image.open(path) as image:
                image.verify()
    report["frozen_eval"] = dict(
        pairs=len(groups),
        episodes=sum(map(len, groups.values())),
        families=dict(Counter((ts[0]["benchmark_family"] for ts in groups.values()))),
    )
    report["unique_referenced_images"] = len(images)
    report["images_decoded"] = verify_images
    return report
