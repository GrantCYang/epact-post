"""SCHEMA POST analytic scorer (design v1.0 §6). SELF-TESTING: refuses
to run if vectors fail. Re-parses choices with the frozen parser.

Evidence semantics (sound under the body rules, so an entailed choice
always equals the label when defined):
  feasible(o):   a successful pickup of o anywhere (picked => mass<=mu,
                 and reachable from where the body actually was).
  infeasible(o): the mu line on o (too heavy), OR the rho line on o
                 while the same episode also observed the motors line
                 (out of reach and cannot walk).
  entailed choice: the heaviest candidate with feasible evidence, if
                 every strictly heavier candidate has infeasible
                 evidence; else None (UNDETERMINED).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from post.score import parse_choice

from post.evidence_rule import evidence


def _selftest() -> None:
    C = [
        {"name": "d", "mass": 0.1},
        {"name": "n", "mass": 0.4},
        {"name": "h", "mass": 1.0},
    ]
    A = lambda nm, rule=None, ok=False: {
        "action": {"try_lift": nm},
        "log": {"ok": ok, "rule": rule},
    }
    M = {"action": {"move": "forward"}, "log": {"ok": False, "rule": "legs_dead"}}
    if not evidence([A("h", ok=True)], C)["entailed"] == "h":
        raise ValueError("Numerical validation failed at source line 61")
    if not evidence([A("h", rule="mu"), A("n", ok=True)], C)["entailed"] == "n":
        raise ValueError("Numerical validation failed at source line 62")
    if not evidence([A("n", ok=True)], C)["entailed"] is None:
        raise ValueError("Numerical validation failed at source line 64")
    if not evidence([M, A("h", rule="rho"), A("n", ok=True)], C)["entailed"] == "n":
        raise ValueError("Numerical validation failed at source line 65")
    if not evidence([A("h", rule="rho"), A("n", ok=True)], C)["entailed"] is None:
        raise ValueError("Numerical validation failed at source line 67")
    if not evidence([], C)["entailed"] is None:
        raise ValueError("Numerical validation failed at source line 69")


_selftest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", type=Path, required=True)
    a = ap.parse_args()
    rows, by_pair = ([], {})
    for f in sorted((a.cell_dir / "episodes").glob("*.json")):
        ep = json.loads(f.read_text())
        names = tuple((c["name"] for c in ep["candidates"]))
        rp = parse_choice(ep["choice_reply"] or "", names)
        ev = evidence(ep.get("actions") or [], ep["candidates"])
        kinds: dict = {}
        for act in ep.get("actions") or []:
            k = next(iter(act["action"]))
            kinds[k] = kinds.get(k, 0) + 1
        headings = {act["agent"]["rotation"] for act in ep.get("actions") or []}
        row = {
            "pair_id": ep["pair_id"],
            "dim": ep["dim"],
            "member": ep["member"],
            "choice": rp["choice"],
            "label": ep["label_choice"],
            "correct": rp["choice"] == ep["label_choice"] if rp["choice"] else None,
            "parse_match": rp["choice"]
            == (ep.get("parsed_choice") or {}).get("choice"),
            "n_actions": len(ep.get("actions") or []),
            "kinds": kinds,
            "headings_seen": len(headings),
            "entailed": ev["entailed"],
            "motors_observed": ev["motors_observed"],
            "feas_evidence": ev["feas"],
            "choice_matches_entailed": rp["choice"] == ev["entailed"]
            if ev["entailed"]
            else None,
            "early_done": any(
                (
                    (t.get("parsed") or {}).get("kind") == "done"
                    for t in ep.get("turns") or []
                )
            ),
            "invalid_turns": sum(
                (
                    1
                    for t in ep.get("turns") or []
                    if (t.get("parsed") or {}).get("kind") == "invalid"
                )
            ),
            "delivered": (ep.get("delivery") or {}).get("delivered"),
            "error": ep.get("ERROR"),
        }
        rows.append(row)
        by_pair.setdefault(ep["pair_id"], {})[ep["member"]] = row
    dims = sorted({r["dim"] for r in rows})
    per_dim = {}
    for d in dims:
        ps = {
            pid: m
            for pid, m in by_pair.items()
            if any((r["dim"] == d for r in m.values()))
        }
        flips = sum(
            (
                1
                for m in ps.values()
                if m.get("A", {}).get("correct") and m.get("B", {}).get("correct")
            )
        )
        same = sum(
            (
                1
                for m in ps.values()
                if m.get("A", {}).get("choice")
                and m.get("A", {}).get("choice") == m.get("B", {}).get("choice")
            )
        )
        per_dim[d] = {
            "pairs": len(ps),
            "pair_flip": flips,
            "same_choice": same,
            "per_member": sum(
                (1 for m in ps.values() for r in m.values() if r["correct"])
            ),
        }
    summary = {
        "cell_dir": str(a.cell_dir),
        "n_episodes": len(rows),
        "per_dim": per_dim,
        "pair_flip_total": sum((v["pair_flip"] for v in per_dim.values())),
        "members_entailed": sum((1 for r in rows if r["entailed"])),
        "choice_matched_entailed": sum(
            (1 for r in rows if r["choice_matches_entailed"])
        ),
        "mean_actions": round(
            sum((r["n_actions"] for r in rows)) / max(1, len(rows)), 2
        ),
        "invalid_turns": sum((r["invalid_turns"] for r in rows)),
        "early_done": sum((1 for r in rows if r["early_done"])),
        "parse_mismatches": sum((1 for r in rows if not r["parse_match"])),
        "errors": sum((1 for r in rows if r["error"])),
        "rows": rows,
    }
    (a.cell_dir / "report_post.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    for r in rows:
        print(
            f"  {r['pair_id']}/{r['member']}: choice={r['choice']} label={r['label']} correct={r['correct']} acts={r['n_actions']} ent={r['entailed']} kinds={r['kinds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
