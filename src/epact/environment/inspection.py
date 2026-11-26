"""Versioned inspection interface: grounded query references and usable gaze.

The references identify entities already specified by the query. CHECK's
queried_object is NOT referenceable in presence questions. Policy inputs
never contain hidden labels, object metadata, or evidence flags.
"""

from __future__ import annotations
import math
import copy
from epact.environment.actions import (
    ActionAdapter,
    ActionResult,
    UNSEEN,
    _clamp_horizon,
    _yaw_towards,
)
from epact.environment.episode import Episode
from epact.environment.sim import dist_xz, EpactSim, controller_types

PROTOCOL = "inspection-v4-dev"
PUBLIC_HELP = 'Use target="query_location" for the container/surface named in a presence question, and target="query_object" for the object named in a state question. For presence questions, inspect query_location; the queried item itself cannot be an action target. You may also use object type names such as "Fridge" or door IDs returned in action errors. Do not invent instance IDs. GoTo accepts view="near" (default, for opening) or view="inspect" (medium distance), or view="wide" (fuller view), and optional side="north"/"east"/"south"/"west". GoTo also accepts stance="stand" (default) or "crouch" to inspect low shelves. LookAt accepts region="center" (default), "upper", or "lower". A state question asks about the existing state: inspect it without changing it. If the room is dark, target="room_light" refers to its available light control.'


class InspectionSim(EpactSim):
    def start(self, house):
        Controller, CloudRendering = controller_types()
        self.house = house
        self.controller = Controller(
            platform=CloudRendering,
            scene=house,
            local_executable_path=str(self.executable_path),
            width=self.width,
            height=self.height,
            fieldOfView=90,
            quality="Low",
            visibilityDistance=self.VISIBILITY_M,
            renderDepthImage=False,
            renderInstanceSegmentation=True,
        )


def bounds(obj):
    box = obj.get("axisAlignedBoundingBox") or {}
    return (
        box.get("center", obj["position"]),
        box.get("size", {"x": 0.2, "y": 0.2, "z": 0.2}),
    )


def aim_point(obj, region="center"):
    center, size = bounds(obj)
    point = dict(center)
    surface = obj.get("objectType") in {
        "DiningTable",
        "CoffeeTable",
        "SideTable",
        "Desk",
        "Dresser",
        "CounterTop",
        "SinkBasin",
    }
    fraction = {"upper": 0.35, "lower": -0.35, "center": 0.42 if surface else 0}[region]
    point["y"] += fraction * size["y"]
    return point


class InspectionAdapter(ActionAdapter):
    def __init__(self, sim, task):
        super().__init__(sim)
        self.task = task
        self.pose_cache = {}
        if task["query_type"] == "STATE":
            self.referenceable.add(task["target_object_id"])

    def resolve_target(self, target):
        key = str(target).strip().lower()
        if self.task["query_type"] == "CHECK":
            import re

            cleaned = re.sub("[^a-z0-9]", "", key)
            target_type = re.sub(
                "[^a-z0-9]", "", self.task.get("target_type", "").lower()
            )
            if key == "query_object" or (target_type and cleaned == target_type):
                return UNSEEN
            if str(target) == self.task.get("target_object_id"):
                return UNSEEN
        darkness = (self.task.get("house_mod") or {}).get("darkness") or {}
        lamps = darkness.get("lit_when_any_on") or []
        light_off = bool(lamps) and (
            not any(((self.sim.get_object(i) or {}).get("isToggled") for i in lamps))
        )
        if key == "room_light":
            return self.task.get("lamp_id") or (lamps[0] if lamps else UNSEEN)
        if light_off and self.task["query_type"] == "CHECK":
            if key == "query_object":
                return UNSEEN
            import re

            cleaned = re.sub("[^a-z0-9]", "", key)
            target_type = re.sub(
                "[^a-z0-9]", "", self.task.get("target_type", "").lower()
            )
            if cleaned == target_type or str(target) == self.task.get(
                "target_object_id"
            ):
                return UNSEEN
        if key == "query_location":
            return (
                self.task.get("receptacle_id")
                or self.task.get("target_object_id")
                or UNSEEN
            )
        if key == "query_object":
            oid = self.task.get("target_object_id")
            obj = self.sim.get_object(oid) if oid else None
            if obj and (self.task["query_type"] == "STATE" or obj.get("visible")):
                return oid
            return UNSEEN
        resolved = super().resolve_target(target)
        resolved_obj = self.sim.get_object(resolved) if resolved != UNSEEN else None
        if self.task["query_type"] == "CHECK" and resolved_obj:
            if resolved_obj.get("objectType") == self.task.get("target_type"):
                return UNSEEN
        return resolved

    def execute(self, name, **args):
        result = super().execute(name, **args)
        result.cost = 1.0
        if "out of reach" in result.error:
            result.error = "target is out of reach; move closer"
        if result.ok and name in {"Open", "Close", "Toggle", "MoveAside"}:
            self.pose_cache.clear()
        return result

    def _goto(self, target, view="near", side="auto", stance="stand"):
        if (
            view not in {"near", "inspect", "wide"}
            or side not in {"auto", "north", "east", "south", "west"}
            or stance not in {"stand", "crouch"}
        ):
            return ActionResult(
                "GoTo",
                {"target": target},
                False,
                error="invalid view, side, or stance; use the documented options",
            )
        self.requested_view, self.requested_side = (view, side)
        self.requested_stance = stance
        result = super()._goto(target)
        result.args.update(view=view, side=side, stance=stance)
        if result.ok and self.sim.get_object(target):
            gaze = self._look_at(target)
            result.native_steps.extend(gaze.native_steps)
            result.detail["arrival_gaze"] = gaze.detail
            if not gaze.ok:
                result.detail["arrival_gaze_error"] = gaze.error
        return result

    def _object_pose(self, object_id, obj, result):
        obj = self.inspection_object(obj)
        key = (
            object_id,
            tuple((round(obj["position"][k], 4) for k in ("x", "y", "z"))),
            obj.get("isOpen"),
            obj.get("isToggled"),
        )
        poses = self.pose_cache.get(key)
        ok = True
        if poses is None:
            event, ok = self._step(
                result, action="GetInteractablePoses", objectId=object_id
            )
            poses = event.metadata.get("actionReturn") or []
            if ok:
                self.pose_cache[key] = poses
        if not ok or not poses:
            result.error = "no reachable inspection pose"
            return None
        center, size = bounds(obj)
        view = getattr(self, "requested_view", "near")
        side = getattr(self, "requested_side", "auto")
        desired = (
            max(1.65, 0.95 * size["y"], 0.7 * size["x"], 0.7 * size["z"])
            if view == "wide"
            else 1.3
            if view == "inspect"
            else 1.0
        )
        compass = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}

        def score(p):
            distance = dist_xz(p, center)
            side_error = 0.0
            if side != "auto":
                dx, dz = compass[side]
                side_error = math.hypot(
                    p["x"] - (center["x"] + desired * dx),
                    p["z"] - (center["z"] + desired * dz),
                )
            wanted_standing = getattr(self, "requested_stance", "stand") == "stand"
            return (
                p.get("standing", True) != wanted_standing,
                round(side_error, 3),
                round(abs(distance - desired), 3),
                round(p["x"], 4),
                round(p["z"], 4),
                p.get("rotation", 0),
                p.get("horizon", 0),
            )

        p = min(poses, key=score)
        point = aim_point(obj)
        eye = {
            "x": p["x"],
            "y": p["y"] + (0.675 if p.get("standing", True) else 0.0),
            "z": p["z"],
        }
        pitch = math.degrees(
            math.atan2(eye["y"] - point["y"], max(dist_xz(eye, point), 1e-06))
        )
        return {
            "position": {k: p[k] for k in ("x", "y", "z")},
            "rotation": round(_yaw_towards(eye, point), 2),
            "horizon": _clamp_horizon(pitch),
            "standing": p.get("standing", True),
        }

    def _look_at(self, target, region="center"):
        result = ActionResult("LookAt", {"target": target, "region": region}, False)
        if region not in {"center", "upper", "lower"}:
            result.error = "region must be center/upper/lower"
            return result
        obj = self.sim.get_object(target)
        if not obj:
            result.error = "target not in view; explore to locate it"
            return result
        obj = self.inspection_object(obj)
        agent = self.sim.agent_pose()
        here = self.sim.room_of_point(agent["position"]["x"], agent["position"]["z"])
        there = self.sim.room_of_point(obj["position"]["x"], obj["position"]["z"])
        if here is not None and there is not None and (here != there):
            result.error = "target is in another room; navigate there first"
            return result
        camera = self.sim.metadata.get("cameraPosition")
        if not camera:
            raise RuntimeError(
                "cameraPosition missing; refusing an unverified gaze estimate"
            )
        point = aim_point(obj, region)
        yaw = _yaw_towards(camera, point)
        pitch = math.degrees(
            math.atan2(camera["y"] - point["y"], max(dist_xz(camera, point), 1e-06))
        )
        _, ok = self._step(
            result,
            action="Teleport",
            position=agent["position"],
            rotation={"x": 0, "y": round(yaw, 2), "z": 0},
            horizon=_clamp_horizon(pitch),
            standing=agent["standing"],
        )
        result.ok = ok
        result.cost = 0.5 if ok else 0
        result.detail = {
            "camera": camera,
            "aim": point,
            "yaw": yaw,
            "horizon": _clamp_horizon(pitch),
        }
        if not ok:
            result.error = result.native_steps[-1]["error"]
        return result

    def inspection_object(self, obj):
        if obj["objectId"] == self.task.get("receptacle_id") and getattr(
            self, "anchor_bounds", None
        ):
            return {**obj, "axisAlignedBoundingBox": self.anchor_bounds}
        return obj


class InspectionEpisode(Episode):
    def __init__(self, sim, task, dataset_path, save_frames_dir=None):
        super().__init__(sim, {**task, "budget": 20.0}, dataset_path, save_frames_dir)
        self.adapter = InspectionAdapter(sim, self.task)

    def setup(self, house_cache=None):
        super().setup(house_cache)
        self.adapter.pose_cache.clear()
        rec = (
            self.sim.get_object(self.task.get("receptacle_id"))
            if self.task.get("receptacle_id")
            else None
        )
        self.inspection_anchor_bounds = (
            copy.deepcopy(rec.get("axisAlignedBoundingBox")) if rec else None
        )
        self.adapter.anchor_bounds = self.inspection_anchor_bounds

    def act(self, name, **args):
        if name == "Answer":
            try:
                confidence = float(args.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.0
            if not math.isfinite(confidence):
                confidence = 0.0
            args = {**args, "confidence": min(1.0, max(0.0, confidence))}
        return super().act(name, **args)
