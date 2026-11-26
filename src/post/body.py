"""Rule-governed body over the DEFAULT THOR agent (SCHEMA POST v1.0
§2-§3). The engine renders and tracks world state; EVERY action outcome
is decided here by the theta rules. The arm agent is retired (rule-grasp
probe, commit 27046a8: the engine vetoes arm grasps but obeys
forceAction on the default agent unconditionally).

theta = {"legs": "ok"|"dead", "rho": float, "mu": float}
"""

from __future__ import annotations
import math
import sys
from pathlib import Path
from research_common.runtime import require_staged_build

GRID = 0.25
VISIBILITY = 10.0
HORIZON = 30
ADJ = 0.8
L_MOVED = "[move outcome] moved forward 0.25 m."
L_BLOCKED = "[move outcome] blocked by an obstacle; position unchanged."
L_MOTORS = "[move outcome] the drive motors did not respond; position unchanged."
L_PICKED = "[lift outcome] picked up the {name}; now holding it."
L_FAR = "[lift outcome] the {name} is out of reach from here; not picked."
L_HEAVY = "[lift outcome] the {name} did not budge; too heavy for this body."
L_PUTDOWN = "[lift outcome] put down the {name}."
L_REFUSED = "[action refused] {why}. That turn is spent."
FACING = {0: "N", 90: "E", 180: "S", 270: "W"}


def dist2d(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


class PostBody:
    def __init__(self):
        self.c = None
        self.theta: dict | None = None
        self.scene: str | None = None
        self.held: str | None = None

    def launch(self, scene: str) -> None:
        if self.c is None:
            from ai2thor.controller import Controller
            from ai2thor.platform import CloudRendering

            self.c = Controller(
                scene=scene,
                width=448,
                height=448,
                local_executable_path=str(require_staged_build()),
                gridSize=GRID,
                snapToGrid=True,
                rotateStepDegrees=90,
                visibilityDistance=VISIBILITY,
                platform=CloudRendering,
            )
            self.scene = scene
        elif scene != self.scene:
            self.c.reset(scene=scene)
            self.scene = scene
        else:
            self.c.reset(scene=scene)

    def stop(self) -> None:
        if self.c is not None:
            self.c.stop()
            self.c = None

    def md(self) -> dict:
        return self.c.last_event.metadata

    def objects(self) -> list:
        return self.md()["objects"]

    def agent_pose(self) -> dict:
        a = self.md()["agent"]
        return {
            "position": a["position"],
            "rotation": round(a["rotation"]["y"]) % 360,
            "horizon": a["cameraHorizon"],
        }

    def frame_png(self) -> bytes:
        import io
        from PIL import Image

        im = Image.fromarray(self.c.last_event.frame)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def setup_episode(self, spec: dict, theta: dict) -> None:
        """spec: {scene, dock: {position{x,y,z}, rotation_y}, candidates:
        [{name, object_id, mass}, ...]}"""
        self.launch(spec["scene"])
        self.held = None
        self.theta = dict(theta)
        ev = self.c.step(
            action="Teleport",
            position=spec["dock"]["position"],
            rotation={"x": 0, "y": spec["dock"]["rotation_y"], "z": 0},
            horizon=HORIZON,
            standing=True,
        )
        if not ev.metadata["lastActionSuccess"]:
            raise RuntimeError(
                f"Teleport to dock failed: {ev.metadata['errorMessage']}"
            )

    def odometry(self) -> str:
        p = self.agent_pose()
        f = FACING.get(p["rotation"] % 360, str(p["rotation"]))
        return f"[odometry] at ({p['position']['x']:.2f}, {p['position']['z']:.2f}), facing {f}."

    def visible_by_name(self, name: str) -> dict | None:
        """Nearest currently-visible pickupable object of this type."""
        name = name.strip().lower()
        cands = [
            o
            for o in self.objects()
            if o["pickupable"] and o["visible"] and (o["objectType"].lower() == name)
        ]
        if not cands:
            return None
        return min(cands, key=lambda o: o["distance"])

    def known_name(self, name: str) -> bool:
        name = name.strip().lower()
        return any(
            (
                o["pickupable"] and o["objectType"].lower() == name
                for o in self.objects()
            )
        )

    def act(self, action: dict) -> dict:
        """One checkup action. Returns {"lines": [...], "log": {...}};
        odometry is appended by the caller every turn."""
        log: dict = {"action": action, "theta_applied": None}
        lines: list = []
        kind = next(
            iter(
                (k for k in ("rotate", "move", "try_lift", "put_down") if k in action)
            ),
            None,
        )
        if kind == "rotate":
            d = str(action["rotate"]).strip().lower()
            self.c.step(action="RotateLeft" if d == "left" else "RotateRight")
            log["ok"] = True
        elif kind == "move":
            if self.theta["legs"] == "dead":
                lines.append(L_MOTORS)
                log.update(ok=False, rule="legs_dead", theta_applied="legs")
            else:
                ev = self.c.step(action="MoveAhead")
                ok = ev.metadata["lastActionSuccess"]
                lines.append(L_MOVED if ok else L_BLOCKED)
                log.update(ok=ok, rule="world" if not ok else None)
        elif kind == "try_lift":
            name = str(action["try_lift"]).strip().lower()
            if self.held is not None:
                lines.append(
                    L_REFUSED.format(why="already holding an object; put it down first")
                )
                log.update(ok=False, rule="holding")
            elif not self.known_name(name):
                lines.append(L_REFUSED.format(why=f"no object called '{name}' here"))
                log.update(ok=False, rule="unknown_name")
            else:
                o = self.visible_by_name(name)
                if o is None:
                    lines.append(
                        L_REFUSED.format(
                            why=f"the {name} is not in view; rotate to face it first"
                        )
                    )
                    log.update(ok=False, rule="not_visible")
                else:
                    d = o["distance"]
                    log["target"] = {
                        "object_id": o["objectId"],
                        "mass": o["mass"],
                        "distance": round(d, 3),
                    }
                    if d > self.theta["rho"]:
                        lines.append(L_FAR.format(name=name))
                        log.update(ok=False, rule="rho", theta_applied="rho")
                    elif o["mass"] > self.theta["mu"]:
                        lines.append(L_HEAVY.format(name=name))
                        log.update(ok=False, rule="mu", theta_applied="mu")
                    else:
                        ev = self.c.step(
                            action="PickupObject",
                            objectId=o["objectId"],
                            forceAction=True,
                        )
                        ok = ev.metadata["lastActionSuccess"]
                        if ok:
                            self.held = o["objectId"]
                            lines.append(L_PICKED.format(name=name))
                        else:
                            lines.append(
                                L_REFUSED.format(why=f"the {name} could not be grasped")
                            )
                        log.update(
                            ok=ok,
                            rule=None if ok else "ENGINE_VETO",
                            engine_error=(ev.metadata.get("errorMessage") or "")[:120],
                        )
        elif kind == "put_down":
            if self.held is None:
                lines.append(L_REFUSED.format(why="not holding anything"))
                log.update(ok=False, rule="not_holding")
            else:
                name = self.held.split("|")[0].lower()
                self.c.step(action="DropHandObject", forceAction=True)
                self.held = None
                lines.append(L_PUTDOWN.format(name=name))
                log.update(ok=True)
        else:
            lines.append(L_REFUSED.format(why="not a valid action"))
            log.update(ok=False, rule="invalid")
        return {"lines": lines, "log": log}

    def privileged(self, spec: dict) -> dict:
        ap = self.agent_pose()
        cand = {}
        for cd in spec["candidates"]:
            o = next(
                (o for o in self.objects() if o["objectId"] == cd["object_id"]), None
            )
            if o:
                cand[cd["name"]] = {
                    "distance": round(o["distance"], 3),
                    "visible": o["visible"],
                    "mass": o["mass"],
                    "position": o["position"],
                }
        return {"agent": ap, "held": self.held, "theta": self.theta, "candidates": cand}

    def reachable(self) -> list:
        ev = self.c.step(action="GetReachablePositions")
        return ev.metadata["actionReturn"]
