"""CPU episode server: JSON-lines over localhost, explicit task and house inputs.

Start/act/outcome messages expose public observations and separate scoring flags.
The describe message binds task/dataset identities before model execution."""

from __future__ import annotations
import argparse
import base64
import io
import json
import socketserver
import sys
from pathlib import Path
from epact.data.paths import house_dataset
from epact.environment.sim import frame_hash

ACTIONS_SCHEMA = [
    {"name": "GoTo", "args": {"target": "room id or object id"}},
    {"name": "Open", "args": {"target": "object id"}},
    {"name": "Close", "args": {"target": "object id"}},
    {"name": "Toggle", "args": {"target": "object id"}},
    {"name": "MoveAside", "args": {"target": "object id"}},
    {"name": "LookAt", "args": {"target": "object id"}},
    {"name": "Answer", "args": {"answer": "yes|no", "confidence": "0..1"}},
]


def frame_b64(episode) -> str:
    from PIL import Image

    frame = episode.frame()
    image = Image.fromarray(frame)
    if episode.save_frames_dir is not None:
        path = episode.save_frames_dir / f"{frame_hash(frame)}.png"
        if not path.exists():
            image.save(path)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                msg = json.loads(line)
                reply = self.dispatch(server, msg)
            except Exception as exc:
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()
            if isinstance(msg, dict) and msg.get("cmd") == "shutdown":
                server._BaseServer__shutdown_request = True
                return

    def dispatch(self, server, msg: dict) -> dict:
        cmd = msg.get("cmd")
        if cmd == "describe":
            return {"ok": True, "protocol": server.protocol, **server.input_identity}
        if cmd == "ping":
            return {"ok": True}
        if cmd == "list":
            return {"ok": True, "task_ids": list(server.tasks.keys())}
        if cmd == "start":
            task = server.tasks[msg["task_id"]]
            server.episode = server.episode_class(
                server.sim, task, server.dataset, save_frames_dir=server.save_frames_dir
            )
            server.episode.setup(server.house_cache)
            return {
                "ok": True,
                "task_id": task["task_id"],
                "query": task["query"],
                "budget": server.episode.task["budget"],
                "protocol": server.protocol,
                "reference_help": server.reference_help,
                "actions_schema": ACTIONS_SCHEMA,
                "frame_png_b64": frame_b64(server.episode),
                "frame_sha256": frame_hash(server.episode.frame()),
                "privileged_determinable": server.episode.determinable(),
                "privileged_evidence_proxy": server.evidence_assessor(server.episode),
            }
        if cmd == "act":
            episode = server.episode
            out = episode.act(msg["action"], **msg.get("args", {}))
            reply = {"ok": True, "terminated": out["terminated"]}
            if not out["terminated"]:
                reply["result"] = {k: out["result"][k] for k in ("ok", "error", "cost")}
                reply["frame_png_b64"] = frame_b64(episode)
                reply["frame_sha256"] = frame_hash(episode.frame())
                reply["remaining_budget"] = round(
                    episode.task["budget"] - episode.cost, 3
                )
                reply["privileged_determinable"] = episode.determinable()
                reply["privileged_evidence_proxy"] = server.evidence_assessor(episode)
            else:
                reply["outcome"] = episode.outcome
                reply["outcome"]["evidence_proxy_at_termination"] = (
                    server.evidence_assessor(episode)
                )
                reply["outcome"]["step_log"] = episode.steps
            return reply
        if cmd == "outcome":
            return {"ok": True, "outcome": server.episode.outcome}
        if cmd == "shutdown":
            return {"ok": True}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="ProcTHOR split file; defaults to EPACT_HOUSE_DATASET or data/external/procthor-10k/<split>.jsonl.gz",
    )
    parser.add_argument("--port", type=int, default=7301)
    parser.add_argument("--save-frames-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--protocol", choices=("legacy-v3", "inspection-v4-dev"), default="legacy-v3"
    )
    args = parser.parse_args()
    args.dataset = house_dataset(args.dataset, args.split)
    from epact.environment.episode import Episode
    from epact.environment.sim import EpactSim

    tasks = {}
    with (args.data_root / f"{args.split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            tasks[task["task_id"]] = task
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as server:
        from research_common.run_state import sha256

        server.input_identity = {
            "tasks_sha256": sha256(args.data_root / f"{args.split}.jsonl"),
            "dataset_sha256": sha256(args.dataset),
        }
        server.tasks = tasks
        server.dataset = args.dataset
        server.protocol = args.protocol
        server.episode_class = Episode
        server.reference_help = ""
        server.evidence_assessor = lambda ep: None
        sim_class = EpactSim
        if args.protocol == "inspection-v4-dev":
            from epact.environment.inspection import (
                InspectionEpisode,
                InspectionSim,
                PUBLIC_HELP,
            )
            from epact.environment.evidence import assess

            server.episode_class = InspectionEpisode
            server.reference_help = PUBLIC_HELP
            server.evidence_assessor = assess
            sim_class = InspectionSim
        server.sim = sim_class(width=args.width, height=args.height)
        if args.save_frames_dir:
            args.save_frames_dir.mkdir(parents=True, exist_ok=True)
        server.save_frames_dir = args.save_frames_dir
        server.house_cache = {}
        server.episode = None
        print(
            f"[episode-server] {len(tasks)} tasks on 127.0.0.1:{args.port}", flush=True
        )
        try:
            server.serve_forever()
        finally:
            server.sim.stop()


if __name__ == "__main__":
    main()
