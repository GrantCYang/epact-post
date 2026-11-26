"""POST simulator server: body rules and observations over a local JSON-lines connection."""

from __future__ import annotations
import argparse
import base64
import json
import socketserver
import sys
from pathlib import Path
from post.body import PostBody, ADJ, dist2d
from post.build import bfs_path, walk, face

BODY = PostBody()
PAIRS: dict = {}
REACH: dict = {}
CUR: dict = {"pair": None, "member": None}


def frame_b64() -> str:
    return base64.b64encode(BODY.frame_png()).decode()


def handle_cmd(msg: dict) -> dict:
    cmd = msg.get("cmd")
    if cmd == "ping":
        return {"ok": True}
    if cmd == "start":
        pair = PAIRS[msg["pair_id"]]
        mem = next((m for m in pair["members"] if m["member"] == msg["member"]))
        spec = {
            "scene": pair["scene"],
            "dock": pair["dock"],
            "candidates": pair["candidates"],
        }
        BODY.setup_episode(spec, mem["theta"])
        CUR["pair"], CUR["member"] = (pair, mem)
        CUR["cand_pos"] = {
            c["name"]: next(
                (
                    o["position"]
                    for o in BODY.objects()
                    if o["objectId"] == c["object_id"]
                )
            )
            for c in pair["candidates"]
        }
        return {
            "ok": True,
            "frame_png_b64": frame_b64(),
            "odometry": BODY.odometry(),
            "privileged": BODY.privileged(spec),
        }
    if cmd == "act":
        pair = CUR["pair"]
        spec = {
            "scene": pair["scene"],
            "dock": pair["dock"],
            "candidates": pair["candidates"],
        }
        r = BODY.act(msg["action"])
        return {
            "ok": True,
            "lines": r["lines"],
            "odometry": BODY.odometry(),
            "frame_png_b64": frame_b64(),
            "privileged": {**BODY.privileged(spec), "log": r["log"]},
        }
    if cmd == "restore":
        pair = CUR["pair"]
        c = next((c for c in pair["candidates"] if c["name"] == msg["name"]))
        BODY.c.step(
            action="TeleportObject",
            objectId=c["object_id"],
            position=CUR["cand_pos"][msg["name"]],
            rotation={"x": 0, "y": 0, "z": 0},
            forceAction=True,
        )
        return {"ok": True}
    if cmd == "render":
        pair = CUR["pair"]
        spec = {
            "scene": pair["scene"],
            "dock": pair["dock"],
            "candidates": pair["candidates"],
        }
        return {
            "ok": True,
            "frame_png_b64": frame_b64(),
            "odometry": BODY.odometry(),
            "privileged": BODY.privileged(spec),
        }
    if cmd == "deliver":
        pair, mem = (CUR["pair"], CUR["member"])
        theta = mem["theta"]
        cand = next((c for c in pair["candidates"] if c["name"] == msg["name"]), None)
        if cand is None:
            return {"ok": True, "delivered": False, "why": "not a candidate"}
        reach = REACH[pair["scene"]]
        dock = pair["dock"]["position"]
        if BODY.held is not None:
            BODY.act({"put_down": True})
        o = BODY.visible_by_name(cand["name"])
        log = []
        if not (o and o["distance"] <= theta["rho"] and (o["mass"] <= theta["mu"])):
            if theta["legs"] == "ok":
                path = None
                for adj_r in (0.5, 0.65, ADJ):
                    path = bfs_path(
                        reach,
                        BODY.agent_pose()["position"],
                        (o or {"position": cand_pos(pair, cand)})["position"]
                        if o
                        else cand_pos(pair, cand),
                        adj_r,
                    )
                    if path:
                        break
                if not path or not walk(BODY, path):
                    return {
                        "ok": True,
                        "delivered": False,
                        "why": "no path",
                        "log": log,
                    }
                for _ in range(4):
                    o = BODY.visible_by_name(cand["name"])
                    if o and o["distance"] <= theta["rho"]:
                        break
                    BODY.c.step(action="RotateRight")
        r = BODY.act({"try_lift": cand["name"]})
        log.append(r["log"])
        if not r["log"].get("ok"):
            return {
                "ok": True,
                "delivered": False,
                "why": r["log"].get("rule"),
                "log": log,
            }
        back = bfs_path(reach, BODY.agent_pose()["position"], dock, 0.2)
        if back is not None and walk(BODY, back):
            arrived = dist2d(BODY.agent_pose()["position"], dock) <= 0.3
            return {"ok": True, "delivered": bool(arrived), "log": log}
        return {
            "ok": True,
            "delivered": theta["legs"] != "ok"
            and dist2d(BODY.agent_pose()["position"], dock) <= 0.3,
            "log": log,
        }
    raise ValueError(f"unknown cmd {cmd}")


def cand_pos(pair: dict, cand: dict) -> dict:
    for o in BODY.objects():
        if o["objectId"] == cand["object_id"]:
            return o["position"]
    raise KeyError(cand["name"])


class H(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for line in self.rfile:
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg.get("cmd") == "shutdown":
                self.wfile.write(b'{"ok": true}\n')
                self.server._BaseServer__shutdown_request = True
                return
            try:
                rep = handle_cmd(msg)
            except Exception as e:
                rep = {"ok": False, "error": repr(e)[:300]}
            self.wfile.write((json.dumps(rep) + "\n").encode())
            self.wfile.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument(
        "--catalog", type=Path, help="Defaults to catalog_post.json next to --pairs"
    )
    a = ap.parse_args()
    cat = json.loads((a.catalog or a.pairs.parent / "catalog_post.json").read_text())
    for line in a.pairs.read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            PAIRS[p["pair_id"]] = p
            REACH[p["scene"]] = cat[p["scene"]]["reachable"]
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", a.port), H)
    print(f"[post-server] {len(PAIRS)} pairs on 127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
