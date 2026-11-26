"""EPACT action library (plan §4) over native AI2-THOR primitives.

Benchmark actions:
    GoTo(x)        cost 1 + 0.1 × path length (m)   x: room id or objectId
    Open(r)        cost 1
    Close(r)       cost 1
    Toggle(s)      cost 1
    MoveAside(o)   cost 2
    LookAt(x)      cost 0.5                          x within current room
    Answer(a, c)   terminates (episode layer, not here)

Every candidate choice (pose, placement surface, path) is made with sorted
deterministic tie-breaks so identical state + action gives identical result.
Failed actions charge no cost and return the error string for the model.
"""

from __future__ import annotations
import heapq
import math
import re
from dataclasses import dataclass, field
from epact.environment.sim import EpactSim, dist_xz, polyline_length

HORIZON_MIN, HORIZON_MAX = (-30.0, 60.0)
GRID = 0.25
REACH_M = 1.5
UNSEEN = "__unseen__"


@dataclass
class ActionResult:
    action: str
    args: dict
    ok: bool
    error: str = ""
    cost: float = 0.0
    detail: dict = field(default_factory=dict)
    native_steps: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "action": self.action,
            "args": self.args,
            "ok": self.ok,
            "error": self.error,
            "cost": round(self.cost, 3),
            "detail": self.detail,
            "native_steps": self.native_steps,
        }


class ActionAdapter:
    def __init__(self, sim: EpactSim) -> None:
        self.sim = sim
        self.referenceable: set[str] = set()

    def execute(self, name: str, **args) -> ActionResult:
        handlers = {
            "GoTo": self._goto,
            "Close": self._close,
            "Open": self._open,
            "Toggle": self._toggle,
            "MoveAside": self._move_aside,
            "LookAt": self._look_at,
        }
        if name not in handlers:
            return ActionResult(name, args, ok=False, error=f"unknown action: {name}")
        if "target" in args and isinstance(args["target"], str):
            resolved = self.resolve_target(args["target"])
            if resolved == UNSEEN:
                return ActionResult(
                    name,
                    args,
                    ok=False,
                    error="target not in view; explore to locate it",
                )
            args = {**args, "target": resolved}
        try:
            return handlers[name](**args)
        except TypeError as exc:
            return ActionResult(name, args, ok=False, error=f"bad arguments: {exc}")

    def resolve_target(self, target: str) -> str:
        """Ground a model-provided name onto an instance id.

        Models see images, not ids, so "Fridge" or "the desk lamp in the
        bedroom" must resolve. Exact object/room ids pass through; otherwise
        match object types (visible instances first, then nearest), then
        room types. Deterministic tie-breaks throughout.
        """
        obj = self.sim.get_object(target)
        if obj is not None:
            if (
                obj.get("pickupable")
                and (not obj.get("visible"))
                and (target not in self.referenceable)
            ):
                return UNSEEN
            return target
        room_ids = {r["id"] for r in self.sim.rooms()}
        if target in room_ids:
            return target
        cleaned = re.sub("[^a-z0-9]", "", target.lower())
        if not cleaned:
            return target
        agent_pos = self.sim.agent_pose()["position"]
        matches = [
            o
            for o in self.sim.objects()
            if re.sub("[^a-z0-9]", "", o["objectType"].lower()) == cleaned
            or cleaned in re.sub("[^a-z0-9]", "", o["objectId"].lower())
        ]
        if matches:
            matches.sort(
                key=lambda o: (
                    not o["visible"],
                    round(dist_xz(o["position"], agent_pos), 4),
                    o["objectId"],
                )
            )
            best = matches[0]
            if (
                best.get("pickupable")
                and (not best.get("visible"))
                and (best["objectId"] not in self.referenceable)
                and (cleaned not in self.referenceable)
            ):
                return UNSEEN
            return best["objectId"]
        for room in sorted(self.sim.rooms(), key=lambda r: r["id"]):
            if re.sub("[^a-z0-9]", "", room.get("roomType", "").lower()) == cleaned:
                return room["id"]
        return UNSEEN

    def _step(self, result: ActionResult, **action):
        event = self.sim.controller.step(**action)
        ok = bool(event.metadata["lastActionSuccess"])
        result.native_steps.append(
            {
                "action": action.get("action"),
                "objectId": action.get("objectId"),
                "success": ok,
                "error": event.metadata.get("errorMessage", ""),
            }
        )
        return (event, ok)

    def _goto(self, target: str) -> ActionResult:
        result = ActionResult("GoTo", {"target": target}, ok=False)
        start = self.sim.agent_pose()["position"]
        room_ids = {r["id"] for r in self.sim.rooms()}
        blocked = self._door_gate(target, start, room_ids)
        if blocked is not None:
            result.error = blocked
            return result
        if target in room_ids:
            pose = self._room_pose(target, result)
        else:
            obj = self.sim.get_object(target)
            if obj is None:
                result.error = f"precondition: no room or object named {target!r}"
                return result
            pose = self._object_pose(target, obj, result)
        if pose is None:
            if not result.error:
                result.error = f"no reachable pose for {target!r}"
            return result
        path_m, path_method = self._path_length(start, pose["position"], result)
        event, ok = self._step(
            result,
            action="Teleport",
            position=pose["position"],
            rotation={"x": 0, "y": pose["rotation"], "z": 0},
            horizon=pose["horizon"],
            standing=pose["standing"],
        )
        if not ok:
            result.error = result.native_steps[-1]["error"] or "Teleport failed"
            return result
        result.ok = True
        result.cost = 1.0 + 0.1 * path_m
        result.detail = {
            "path_length_m": round(path_m, 3),
            "path_method": path_method,
            "pose": pose,
        }
        return result

    def _door_gate(self, target: str, start: dict, room_ids: set) -> str | None:
        """GoTo teleports, so closed doors must gate as a precondition:
        every openable door on the room path has to be open. Doors
        themselves are reachable from either of their sides."""
        start_room = self.sim.room_of_point(start["x"], start["z"])
        if start_room is None:
            return None
        if target in room_ids:
            goal_rooms = [target]
        else:
            obj = self.sim.get_object(target)
            if obj is None:
                return None
            if target.startswith("door"):
                for link in self.sim.room_links():
                    if link["id"] == target:
                        goal_rooms = list(link["rooms"])
                        break
                else:
                    goal_rooms = []
            else:
                room = self.sim.room_of_point(
                    obj["position"]["x"], obj["position"]["z"]
                )
                goal_rooms = [room] if room else []
        if not goal_rooms:
            return None
        errors = []
        for goal in goal_rooms:
            path = self.sim.room_path_doors(start_room, goal)
            if path is None:
                return None
            closed = [
                link["id"]
                for link in path
                if link["gated"]
                and link["id"] != target
                and (not (self.sim.get_object(link["id"]) or {}).get("isOpen", True))
            ]
            if not closed:
                return None
            errors.append(f"the door {closed[0]} on the way is closed")
        return f"precondition: {errors[0]}" if errors else None

    def _object_pose(self, object_id: str, obj: dict, result: ActionResult):
        event, ok = self._step(
            result, action="GetInteractablePoses", objectId=object_id
        )
        poses = event.metadata.get("actionReturn") or []
        if not ok or not poses:
            result.error = f"precondition: {object_id!r} has no interactable pose"
            return None
        target_pos = obj["position"]
        best = sorted(
            poses,
            key=lambda p: (
                round(dist_xz(p, target_pos), 4),
                round(p["x"], 4),
                round(p["z"], 4),
                p.get("rotation", 0),
                p.get("horizon", 0),
            ),
        )[0]
        return {
            "position": {"x": best["x"], "y": best["y"], "z": best["z"]},
            "rotation": best.get("rotation", 0),
            "horizon": _clamp_horizon(best.get("horizon", 0)),
            "standing": best.get("standing", True),
        }

    def _room_pose(self, room_id: str, result: ActionResult):
        centroid = self.sim.room_centroid(room_id)
        if centroid is None:
            result.error = f"precondition: unknown room {room_id!r}"
            return None
        event, ok = self._step(result, action="GetReachablePositions")
        positions = event.metadata.get("actionReturn") or []
        if not ok or not positions:
            result.error = "GetReachablePositions failed"
            return None
        cx, cz = centroid
        in_room = [
            p for p in positions if self.sim.room_of_point(p["x"], p["z"]) == room_id
        ]
        if not in_room:
            result.error = f"no reachable position inside room {room_id!r}"
            return None
        best = sorted(
            in_room,
            key=lambda p: (
                round(math.hypot(p["x"] - cx, p["z"] - cz), 4),
                round(p["x"], 4),
                round(p["z"], 4),
            ),
        )[0]
        yaw = _yaw_towards(best, {"x": cx, "z": cz})
        return {
            "position": dict(best),
            "rotation": round(yaw, 2),
            "horizon": 0.0,
            "standing": True,
        }

    def _path_length(self, start: dict, end: dict, result: ActionResult):
        try:
            event, ok = self._step(
                result, action="GetShortestPathToPoint", position=start, target=end
            )
            ret = event.metadata.get("actionReturn")
            if ok and ret and ret.get("corners"):
                return (polyline_length(ret["corners"]), "navmesh")
        except Exception:
            pass
        length = self._grid_path_length(start, end, result)
        if length is not None:
            return (length, "grid_bfs")
        return (dist_xz(start, end), "euclidean")

    def _grid_path_length(self, start: dict, end: dict, result: ActionResult):
        event, ok = self._step(result, action="GetReachablePositions")
        positions = event.metadata.get("actionReturn") or []
        if not ok or not positions:
            return None
        nodes = {(round(p["x"] / GRID), round(p["z"] / GRID)) for p in positions}
        s = _nearest_node(nodes, start)
        e = _nearest_node(nodes, end)
        if s is None or e is None:
            return None
        length = _dijkstra(nodes, s, e)
        return length

    def _open(self, target: str) -> ActionResult:
        return self._simple("Open", target, "OpenObject", "openable")

    def _close(self, target: str) -> ActionResult:
        return self._simple("Close", target, "CloseObject", "openable")

    def _toggle(self, target: str) -> ActionResult:
        result = ActionResult("Toggle", {"target": target}, ok=False)
        obj = self.sim.get_object(target)
        if obj is None:
            result.error = f"precondition: unknown object {target!r}"
            return result
        if not obj["toggleable"]:
            result.error = f"precondition: {target!r} is not toggleable"
            return result
        if obj["distance"] > REACH_M:
            result.error = f"precondition: {target!r} is out of reach ({obj['distance']:.2f} m > {REACH_M} m)"
            return result
        native = "ToggleObjectOff" if obj["isToggled"] else "ToggleObjectOn"
        _, ok = self._step(result, action=native, objectId=target)
        if not ok:
            result.error = result.native_steps[-1]["error"] or f"{native} failed"
            return result
        result.ok = True
        result.cost = 1.0
        result.detail = {"now_toggled": not obj["isToggled"]}
        return result

    def _simple(
        self, name: str, target: str, native: str, capability: str
    ) -> ActionResult:
        result = ActionResult(name, {"target": target}, ok=False)
        obj = self.sim.get_object(target)
        if obj is None:
            result.error = f"precondition: unknown object {target!r}"
            return result
        if not obj[capability]:
            result.error = f"precondition: {target!r} is not {capability}"
            return result
        if obj["distance"] > REACH_M:
            result.error = f"precondition: {target!r} is out of reach ({obj['distance']:.2f} m > {REACH_M} m)"
            return result
        _, ok = self._step(result, action=native, objectId=target)
        if not ok:
            result.error = result.native_steps[-1]["error"] or f"{native} failed"
            return result
        result.ok = True
        result.cost = 1.0
        return result

    def _move_aside(self, target: str) -> ActionResult:
        result = ActionResult("MoveAside", {"target": target}, ok=False)
        obj = self.sim.get_object(target)
        if obj is None:
            result.error = f"precondition: unknown object {target!r}"
            return result
        if not obj["pickupable"]:
            result.error = f"precondition: {target!r} is not pickupable"
            return result
        if obj["distance"] > REACH_M:
            result.error = f"precondition: {target!r} is out of reach ({obj['distance']:.2f} m > {REACH_M} m)"
            return result
        original_parents = list(obj.get("parentReceptacles") or [])
        _, ok = self._step(result, action="PickupObject", objectId=target)
        if not ok:
            result.error = result.native_steps[-1]["error"] or "PickupObject failed"
            return result
        candidates = sorted(
            (
                o
                for o in self.sim.objects()
                if o["receptacle"]
                and o["visible"]
                and (not o["openable"])
                and (o["objectId"] != target)
                and (o["objectId"] not in original_parents)
            ),
            key=lambda o: (round(o["distance"], 4), o["objectId"]),
        )
        placed_on = None
        for candidate in candidates[:5]:
            _, ok = self._step(
                result, action="PutObject", objectId=candidate["objectId"]
            )
            if ok:
                placed_on = candidate["objectId"]
                break
        if placed_on is None:
            _, ok = self._step(result, action="DropHandObject")
            if not ok:
                _, ok = self._step(result, action="DropHandObject", forceAction=True)
            if not ok and original_parents:
                _, ok = self._step(
                    result,
                    action="PutObject",
                    objectId=original_parents[0],
                    forceAction=True,
                )
            if not ok:
                result.error = "could not place, drop, or return the held object"
                return result
            placed_on = "<floor>"
        result.ok = True
        result.cost = 2.0
        result.detail = {"placed_on": placed_on, "moved_from": original_parents}
        return result

    def _look_at(self, target: str) -> ActionResult:
        result = ActionResult("LookAt", {"target": target}, ok=False)
        obj = self.sim.get_object(target)
        if obj is None:
            result.error = f"precondition: unknown object {target!r}"
            return result
        agent = self.sim.agent_pose()
        agent_room = self.sim.room_of_point(
            agent["position"]["x"], agent["position"]["z"]
        )
        obj_room = self.sim.room_of_point(obj["position"]["x"], obj["position"]["z"])
        if agent_room is not None and obj_room is not None and (agent_room != obj_room):
            result.error = (
                f"precondition: {target!r} is in {obj_room}, agent is in {agent_room}"
            )
            return result
        yaw = _yaw_towards(agent["position"], obj["position"])
        eye_height = agent["position"]["y"] + 0.675
        flat = max(dist_xz(agent["position"], obj["position"]), 1e-06)
        pitch = math.degrees(math.atan2(eye_height - obj["position"]["y"], flat))
        _, ok = self._step(
            result,
            action="Teleport",
            position=agent["position"],
            rotation={"x": 0, "y": round(yaw, 2), "z": 0},
            horizon=_clamp_horizon(round(pitch, 2)),
            standing=agent["standing"],
        )
        if not ok:
            result.error = result.native_steps[-1]["error"] or "LookAt teleport failed"
            return result
        result.ok = True
        result.cost = 0.5
        result.detail = {
            "yaw": round(yaw, 2),
            "horizon": _clamp_horizon(round(pitch, 2)),
        }
        return result


def _clamp_horizon(h: float) -> float:
    return max(HORIZON_MIN, min(HORIZON_MAX, h))


def _yaw_towards(src: dict, dst: dict) -> float:
    return math.degrees(math.atan2(dst["x"] - src["x"], dst["z"] - src["z"])) % 360.0


def _nearest_node(nodes: set, point: dict):
    if not nodes:
        return None
    return min(
        nodes,
        key=lambda n: (
            math.hypot(n[0] * GRID - point["x"], n[1] * GRID - point["z"]),
            n,
        ),
    )


def _dijkstra(nodes: set, start, end):
    if start == end:
        return 0.0
    neighbors = [
        (dx, dz, GRID * math.hypot(dx, dz))
        for dx in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dz) != (0, 0)
    ]
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node == end:
            return d
        if d > dist.get(node, math.inf):
            continue
        for dx, dz, w in neighbors:
            nxt = (node[0] + dx, node[1] + dz)
            if nxt not in nodes:
                continue
            nd = d + w
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return None
