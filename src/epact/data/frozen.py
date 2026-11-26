"""Check frozen task suites and safely restore their separately distributed views."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import stat
import zipfile
from research_common.paths import ROOT


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(root=ROOT, images=None, dataset=None):
    base = root / "data/benchmark"
    manifest = json.loads((base / "manifest.json").read_text())
    views = json.loads((base / "start-views.json").read_text())
    requested = set()
    report = {}
    for name, suite in manifest["suites"].items():
        file = root / suite["path"]
        if (
            sha(file) != suite["sha256"]
            or sha(root / suite["pair_file"]) != suite["pair_sha256"]
        ):
            raise ValueError("Frozen task bytes changed: " + name)
        rows = [json.loads(s) for s in file.read_text().splitlines() if s.strip()]
        counts = Counter(r["pair_id"] for r in rows)
        if (
            len(rows) != suite["episodes"]
            or set(counts) != set(suite["pair_ids"])
            or set(counts.values()) != {2}
        ):
            raise ValueError("Frozen pair membership changed: " + name)
        if len({r["task_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate task identifiers")
        if json.loads((root / suite["pair_file"]).read_text()) != suite["pair_ids"]:
            raise ValueError("Pair selection differs")
        requested.update(
            "frames/" + h + ".png" for r in rows for h in r["start_frames"]
        )
        report[name] = {"pairs": len(counts), "episodes": len(rows)}
    if requested != set(views["files"]):
        raise ValueError("Task/image membership differs")
    if images:
        for rel, info in views["files"].items():
            file = images / rel
            if (
                not file.is_file()
                or file.stat().st_size != info["bytes"]
                or sha(file) != info["sha256"]
            ):
                raise ValueError("Missing or changed start view: " + rel)
    if dataset and sha(dataset) != manifest["house_dataset"]["sha256"]:
        raise ValueError("House split differs from preserved input")
    return {
        "suites": report,
        "unique_views": len(requested),
        "images_checked": bool(images),
        "house_input_checked": bool(dataset),
        "rerender_validated": False,
    }


def restore_part(archive, destination, manifest):
    """Validate every entry before any write; extract into a new directory only."""
    parts = [p for p in manifest["parts"] if p["sha256"] == sha(archive)]
    if len(parts) != 1 or archive.stat().st_size != parts[0]["bytes"]:
        raise ValueError("Unknown or damaged image archive")
    part = parts[0]
    if destination.exists():
        raise ValueError("Choose a new output directory")
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(set(names)) != len(names) or set(names) != set(part["members"]) | {
            "NOTICE.txt"
        }:
            raise ValueError("Unexpected archive membership")
        payload = {}
        for info in infos:
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Archive links are not allowed")
            if info.filename == "NOTICE.txt":
                if info.file_size > 4096:
                    raise ValueError("Unexpected notice size")
                continue
            rel = Path(info.filename)
            if (
                rel.is_absolute()
                or ".." in rel.parts
                or "\\" in info.filename
                or ":" in info.filename
                or rel.parts[0] != "frames"
            ):
                raise ValueError("Unsafe archive member")
            expected = manifest["files"][info.filename]
            if info.file_size != expected["bytes"] or info.file_size > 1_000_000:
                raise ValueError("Unexpected image size")
            raw = z.read(info)
            if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                raise ValueError("Image bytes differ")
            payload[rel] = raw
    destination.mkdir(parents=True)
    for rel, raw in payload.items():
        path = destination / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return len(payload)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--images", type=Path)
    c.add_argument("--dataset", type=Path)
    r = sub.add_parser("restore-images")
    r.add_argument("--archives", nargs="+", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument(
        "--suite",
        choices=("development", "target-existing", "target-new"),
        required=True,
    )
    for key in ("dataset", "model-path", "output-dir"):
        plan.add_argument("--" + key, type=Path, required=True)
    plan.add_argument("--port", type=int, default=7301)
    args = p.parse_args(argv)
    if args.command == "check":
        print(json.dumps(check(args.root, args.images, args.dataset), indent=2))
        return
    base = args.root / "data/benchmark"
    if args.command == "restore-images":
        manifest = json.loads((base / "start-views.json").read_text())
        if args.out.exists():
            raise ValueError("Choose a new output directory")
        # Stage each validated part separately; never overwrite an existing tree.
        import tempfile, shutil

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
            stage = Path(temp)
            combined = stage / "combined"
            combined.mkdir()
            for index, archive in enumerate(args.archives):
                piece = stage / str(index)
                restore_part(archive, piece, manifest)
                for source in piece.rglob("*.png"):
                    target = combined / "frames" / source.name
                    target.parent.mkdir(exist_ok=True)
                    if target.exists():
                        raise ValueError("Repeated image part")
                    shutil.copyfile(source, target)
            report = check(args.root, combined)
            shutil.copytree(combined, args.out)
        print(json.dumps(report, indent=2))
        return
    check(args.root, dataset=args.dataset)
    suite = json.loads((base / "manifest.json").read_text())["suites"][args.suite]
    server = [
        "epact",
        "policy",
        "serve",
        "--data-root",
        str(base.resolve()),
        "--split",
        args.suite,
        "--dataset",
        str(args.dataset.resolve()),
        "--protocol",
        "inspection-v4-dev",
        "--port",
        str(args.port),
    ]
    evaluate = [
        "epact",
        "policy",
        "evaluate",
        "--tasks",
        str((args.root / suite["path"]).resolve()),
        "--pair-ids",
        str((args.root / suite["pair_file"]).resolve()),
        "--dataset",
        str(args.dataset.resolve()),
        "--model-path",
        str(args.model_path.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--port",
        str(args.port),
        "--mode",
        "reason",
        "--n-rollouts",
        "1",
    ]
    print(
        json.dumps(
            {
                "server_argv": server,
                "evaluation_argv": evaluate,
                "scope": "One reason-first evaluation over the exact selected tasks. Run the server first. Historical multi-arm recipes remain in configs/epact/reason_runs.json; this plan does not reproduce training.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
