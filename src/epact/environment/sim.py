"""Controller lifecycle and house access for EPACT.

The renderer needs a working Vulkan environment. The staged executable is
resolved from EPACT_SIMULATOR_PATH, or the current user's AI2-THOR cache.
No simulator or model is loaded when importing this module.
"""

from __future__ import annotations
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai2thor.controller import Controller
from research_common.runtime import require_staged_build


def controller_types():
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    return (Controller, CloudRendering)


def load_house(dataset: Path, index: int) -> dict:
    if not dataset.exists():
        raise FileNotFoundError(
            f"ProcTHOR dataset not staged: {dataset}; ask the operator to stage it."
        )
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"house index {index} out of range for {dataset}")


def frame_hash(frame) -> str:
    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode())
    digest.update(frame.tobytes())
    return digest.hexdigest()


class EpactSim:
    """Thin wrapper owning one Controller with EPACT's fixed observation spec.

    visibilityDistance is 10 m so the metadata `visible` flag means "in the
    camera view", which is EPACT's hidden/revealed ground-truth semantics.
    Interaction *reach* (1.5 m) is enforced separately by the action adapter.
    """

    VISIBILITY_M = 10.0

    def __init__(self, width: int = 512, height: int = 512) -> None:
        self.executable_path = require_staged_build()
        self.width = width
        self.height = height
        self.controller: Controller | None = None
        self.house: dict | None = None

    def start(self, house: dict) -> None:
        Controller, CloudRendering = controller_types()
        self.house = house
        self.controller = Controller(
            platform=CloudRendering,
            local_executable_path=str(self.executable_path),
            scene=house,
            width=self.width,
            height=self.height,
            fieldOfView=90,
            quality="Low",
            visibilityDistance=self.VISIBILITY_M,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
        )

    def reset(self, house: dict) -> None:
        if not self.controller is not None:
            raise ValueError("start() first")
        self.house = house
        self.controller.reset(scene=house)

    def stop(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    @property
    def metadata(self) -> dict:
        if not self.controller is not None:
            raise ValueError("Numerical validation failed at source line 96")
        return self.controller.last_event.metadata

    @property
    def frame(self):
        if not self.controller is not None:
            raise ValueError("Numerical validation failed at source line 101")
        return self.controller.last_event.frame

    def objects(self) -> list[dict]:
        return self.metadata["objects"]

    def get_object(self, object_id: str) -> dict | None:
        for obj in self.objects():
            if obj["objectId"] == object_id:
                return obj
        return None

    def agent_pose(self) -> dict:
        agent = self.metadata["agent"]
        return {
            "position": dict(agent["position"]),
            "rotation": dict(agent["rotation"]),
            "horizon": agent["cameraHorizon"],
            "standing": agent.get("isStanding", True),
        }

    def rooms(self) -> list[dict]:
        if not self.house is not None:
            raise ValueError("Numerical validation failed at source line 123")
        return self.house.get("rooms", [])

    def room_of_point(self, x: float, z: float) -> str | None:
        for room in self.rooms():
            polygon = [(p["x"], p["z"]) for p in room["floorPolygon"]]
            if _point_in_polygon(x, z, polygon):
                return room["id"]
        return None

    def room_centroid(self, room_id: str) -> tuple[float, float] | None:
        for room in self.rooms():
            if room["id"] == room_id:
                xs = [p["x"] for p in room["floorPolygon"]]
                zs = [p["z"] for p in room["floorPolygon"]]
                return (sum(xs) / len(xs), sum(zs) / len(zs))
        return None

    def room_links(self) -> list[dict]:
        """Openable doors and permanent openings connecting rooms."""
        if not self.house is not None:
            raise ValueError("Numerical validation failed at source line 145")
        links = []
        for door in self.house.get("doors", []):
            if door.get("openness") == 0 and (not door.get("openable")):
                continue
            links.append(
                {
                    "id": door["id"],
                    "rooms": (door["room0"], door["room1"]),
                    "gated": bool(door.get("openable")),
                }
            )
        return links

    def room_path_doors(self, start: str, goal: str) -> list[dict] | None:
        """Doors crossed on the fewest-rooms path, or None if unconnected."""
        if start == goal:
            return []
        links = self.room_links()
        frontier = [(start, [])]
        seen = {start}
        while frontier:
            room, path = frontier.pop(0)
            for link in links:
                if room not in link["rooms"]:
                    continue
                nxt = link["rooms"][0] if link["rooms"][1] == room else link["rooms"][1]
                if nxt in seen:
                    continue
                new_path = path + [link]
                if nxt == goal:
                    return new_path
                seen.add(nxt)
                frontier.append((nxt, new_path))
        return None


def _point_in_polygon(x: float, z: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            inside = not inside
        j = i
    return inside


def dist_xz(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


def polyline_length(corners: list[dict]) -> float:
    total = 0.0
    for a, b in zip(corners, corners[1:]):
        total += dist_xz(a, b)
    return total
