"""Rank shared witness views by evidence proxies; send uncertain views to visual review.
The bundled final training selection additionally includes reviewed and fallback-reviewed pairs.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from research_common.paths import ROOT

AUDIT = ROOT / "data/epact/artifacts/balanced_train_pool_v1/checks_v1/audit"
PHASES = ["near", "wide", "crouch", "east", "south"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def qualify(view, answer, family, min_target):
    """-> ('accept'|'review'|'reject', reason).

    accept: proxy alone suffices. review: goes to the targeted assistant
    visual pass (single frame). reject: unusable at this phase.
    """
    proxy = view.get("evidence_proxy") or {}
    pixels = proxy.get("target_pixels", 0)
    if family == "STATE|D":
        if proxy.get("ready"):
            return ("accept", "state proxy ready")
        if pixels >= min_target:
            return ("review", f"state target visible ({pixels}px) but proxy not ready")
        return ("reject", f"state target too small ({pixels}px)")
    if answer == "yes":
        if proxy.get("geometric_determinable") and pixels >= min_target:
            return ("accept", f"target visible ({pixels}px)")
        if pixels >= min_target:
            return (
                "review",
                f"target visible ({pixels}px) but not flagged determinable",
            )
        return ("reject", f"target_pixels {pixels} < {min_target}")
    if pixels != 0:
        return ("reject", f"negative member shows {pixels} target pixels")
    if proxy.get("ready"):
        return ("accept", "absence proxy ready")
    return (
        "review",
        f"absence coverage unverified ({proxy.get('reason') or 'not ready'})",
    )


def main():
    global AUDIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", type=Path, default=AUDIT)
    ap.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "data/epact/data/balanced_train_pool_v1/train.jsonl",
    )
    ap.add_argument("--min-target-pixels", type=int, default=100)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "data/epact/artifacts/balanced_train_pool_v1/witness_selection_v1.json",
    )
    args = ap.parse_args()
    AUDIT = args.audit_dir
    report = json.loads((AUDIT / "report.json").read_text())
    selected, review_queue, excluded = ({}, {}, {})
    for pairdir in sorted((AUDIT / "pairs").iterdir()):
        audit = json.loads((pairdir / "audit.json").read_text())
        pid, fam = (audit["pair_id"], audit["family"])
        if audit["status"] != "review_pending":
            excluded[pid] = {
                "family": fam,
                "reason": f"capture status {audit['status']}",
            }
            continue
        if audit.get("initial_render", {}).get("needs_leak_review"):
            excluded[pid] = {
                "family": fam,
                "reason": "initial render flagged for leak review",
            }
            continue
        best = None
        reasons = {}
        for phase in PHASES:
            verdicts = []
            for m in audit["members"]:
                view = m["views"].get(phase)
                verdicts.append(
                    qualify(view, m["answer"], fam, args.min_target_pixels)
                    if view
                    else ("reject", "phase missing")
                )
            reasons[phase] = [f"{v}:{r}" for v, r in verdicts]
            if any((v == "reject" for v, _ in verdicts)):
                continue
            rank = sum((v == "review" for v, _ in verdicts))
            if best is None or rank < best[0]:
                best = (rank, phase, verdicts)
            if rank == 0:
                break
        if best is None:
            excluded[pid] = {
                "family": fam,
                "reason": "no phase qualifies both members",
                "phase_reasons": reasons,
            }
            continue
        rank, chosen, verdicts = best
        entry = {
            "family": fam,
            "phase": chosen,
            "members": [
                {
                    "task_id": m["task_id"],
                    "answer": m["answer"],
                    "verdict": v,
                    "verdict_reason": r,
                    "image": m["views"][chosen]["image"],
                    "image_sha256": m["views"][chosen]["image_sha256"],
                    "actions": m["views"][chosen]["actions"],
                    "evidence_proxy": m["views"][chosen]["evidence_proxy"],
                }
                for m, (v, r) in zip(audit["members"], verdicts)
            ],
        }
        if rank == 0:
            selected[pid] = entry
        else:
            review_queue[pid] = entry
    out = {
        "rule": {
            "phases": PHASES,
            "min_target_pixels": args.min_target_pixels,
            "both_members_same_phase": True,
            "complete_pairs_only": True,
        },
        "confessed_limit": "Proxy-filtered, not semantically reviewed. Positive downstream SFT results on the fully reviewed evaluation set remain valid; null results are ambiguous between ineffective supervision and noisy supervision. Spot-sample noise estimate required before reporting.",
        "capture_report": report,
        "auto_selected_pairs": len(selected),
        "review_queue_pairs": len(review_queue),
        "excluded_pairs": len(excluded),
        "family_counts_auto": dict(Counter((v["family"] for v in selected.values()))),
        "family_counts_review": dict(
            Counter((v["family"] for v in review_queue.values()))
        ),
        "family_counts_excluded": dict(
            Counter((v["family"] for v in excluded.values()))
        ),
        "phase_counts": dict(
            Counter(
                (
                    v["phase"]
                    for v in list(selected.values()) + list(review_queue.values())
                )
            )
        ),
        "selected": selected,
        "review_queue": review_queue,
        "excluded": excluded,
        "sources": {p: sha(ROOT / p) for p in ["src/epact/data/selection.py"]},
        "data_sha256": sha(args.tasks),
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "auto_selected_pairs",
                    "review_queue_pairs",
                    "excluded_pairs",
                    "family_counts_auto",
                    "family_counts_review",
                    "phase_counts",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
