"""Auditable image/geometry proxies, kept separate from model correctness.

These checks support visual review; they are not a general proof of semantic
readability. In particular, container coverage needs paired-frame review.
"""

import math

SURFACES = {
    "DiningTable",
    "CoffeeTable",
    "SideTable",
    "Desk",
    "Dresser",
    "CounterTop",
    "SinkBasin",
}
CONTAINERS = {"Fridge", "Microwave", "Cabinet", "Drawer", "Box", "Safe"}


def project(point, camera, yaw, pitch, width=512, height=512, fov=90):
    yaw, pitch = (math.radians(yaw), math.radians(pitch))
    d = {k: point[k] - camera[k] for k in ("x", "y", "z")}
    right = d["x"] * math.cos(yaw) - d["z"] * math.sin(yaw)
    up = (
        d["x"] * math.sin(yaw) * math.sin(pitch)
        + d["y"] * math.cos(pitch)
        + d["z"] * math.cos(yaw) * math.sin(pitch)
    )
    forward = (
        d["x"] * math.sin(yaw) * math.cos(pitch)
        - d["y"] * math.sin(pitch)
        + d["z"] * math.cos(yaw) * math.cos(pitch)
    )
    if forward <= 0.01:
        return None
    scale = height / (2 * math.tan(math.radians(fov) / 2))
    return (width / 2 + scale * right / forward, height / 2 - scale * up / forward)


def corners(box, top_only=False):
    c, s = (box["center"], box["size"])
    if top_only:
        return [
            {
                "x": c["x"] + x * s["x"] / 2,
                "y": c["y"] + s["y"] / 2,
                "z": c["z"] + z * s["z"] / 2,
            }
            for x, z in [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        ]
    return [
        {k: c[k] + sign * s[k] / 2 for k, sign in zip(("x", "y", "z"), signs)}
        for signs in [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    ]


def assess(ep):
    import numpy as np
    from PIL import Image, ImageDraw

    sim, t = (ep.sim, ep.task)
    masks = sim.controller.last_event.instance_masks
    target = (
        sim.get_object(t.get("target_object_id")) if t.get("target_object_id") else None
    )
    anchor_id = (
        t.get("receptacle_id")
        if t["query_type"] == "CHECK"
        else t.get("target_object_id")
    )
    anchor = sim.get_object(anchor_id) if anchor_id else None
    camera, pose = (sim.metadata.get("cameraPosition"), sim.agent_pose())

    def pixels(oid):
        mask = masks.get(oid)
        return int(mask.sum()) if mask is not None else 0

    result = {
        "proxy_version": "inspection-evidence-v1",
        "geometric_determinable": bool(ep.determinable()),
        "target_pixels": pixels(t.get("target_object_id")),
        "anchor_pixels": pixels(anchor_id),
        "camera": camera,
        "pose": pose,
        "ready": False,
        "semantic_review_required": True,
    }
    if not anchor or not camera or (not anchor.get("axisAlignedBoundingBox")):
        result["reason"] = "missing anchor/camera/bounds"
        return result
    box = (
        getattr(ep, "inspection_anchor_bounds", None)
        or anchor["axisAlignedBoundingBox"]
    )
    kind = anchor["objectType"]
    is_surface = kind in SURFACES
    ps = [
        project(
            p, camera, pose["rotation"]["y"], pose["horizon"], sim.width, sim.height
        )
        for p in corners(box, top_only=is_surface)
    ]
    full = all(
        (
            p is not None
            and 8 <= p[0] <= sim.width - 8
            and (8 <= p[1] <= sim.height - 8)
            for p in ps
        )
    )
    result.update(
        anchor_type=kind,
        full_region_in_frame=full,
        is_open=anchor.get("isOpen"),
        above_surface=camera["y"] > box["center"]["y"] + box["size"]["y"] / 2 + 0.05,
    )
    coverage = 0.0
    if is_surface and full:
        region = Image.new("1", (sim.width, sim.height))
        ImageDraw.Draw(region).polygon(ps, fill=1)
        area = np.asarray(region).astype(bool)
        mask = masks.get(anchor_id)
        if mask is not None and area.sum():
            coverage = float((mask & area).sum() / area.sum())
    result["surface_mask_coverage"] = coverage
    if t["query_type"] == "STATE":
        result["ready"] = bool(result["target_pixels"] >= 256 and full)
        result["reason"] = (
            "state target has inspectable image footprint"
            if result["ready"]
            else "state target too small or truncated"
        )
    elif t["answer"] == "yes":
        parents = target.get("parentReceptacles") or [] if target else []
        relation = anchor_id in parents
        result["target_in_queried_receptacle_metadata"] = relation
        result["ready"] = bool(relation and result["target_pixels"] >= 128)
        result["reason"] = (
            "positive target has visible footprint at queried location"
            if result["ready"]
            else "positive target absent/small/wrong location"
        )
    else:
        others = [
            o["objectId"]
            for o in sim.objects()
            if o["objectType"] == t.get("target_type")
            and anchor_id in (o.get("parentReceptacles") or [])
        ]
        result["remaining_same_type_in_location"] = others
        if is_surface:
            result["ready"] = bool(
                not others and full and result["above_surface"] and (coverage >= 0.85)
            )
            result["reason"] = (
                "surface coverage proxy"
                if result["ready"]
                else "surface not fully inspected or label conflict"
            )
        elif kind in CONTAINERS:
            result["ready"] = bool(
                not others
                and full
                and (result["anchor_pixels"] >= 4096)
                and (not anchor.get("openable") or anchor.get("isOpen"))
            )
            result["reason"] = (
                "open container coverage proxy; review interior"
                if result["ready"]
                else "container not fully inspected or label conflict"
            )
        else:
            result["reason"] = "unsupported location semantics"
    return result
