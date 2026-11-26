"""Recompute completed-study statistics from saved scoring inputs, on CPU."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from research_common.paths import ROOT
from research_common.statistics import paired, mcnemar
from epact.evaluation.recorded import opened, episode_summary, witness_scores
from post.recorded import post_summary


def recompute(directory=ROOT / "data/records"):
    payload = (directory / "records.json").read_bytes()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != manifest["records_sha256"]:
        raise ValueError("Evidence checksum mismatch")
    data = json.loads(payload)["datasets"]
    if {k: len(v["rows"]) for k, v in data.items()} != manifest["datasets"]:
        raise ValueError("Evidence coverage mismatch")
    result = {
        "input_sha256": manifest["records_sha256"],
        "episodes": {},
        "witness": {},
        "post": {},
    }
    for key, d in data.items():
        rows = d["rows"]
        if d["kind"] == "episodes":
            result["episodes"][key] = episode_summary(rows)
        elif d["kind"] == "post":
            result["post"][key.removeprefix("post/")] = post_summary(rows)
        else:
            pp, mm = witness_scores(rows)
            entry = {
                "pairs": len(pp),
                "pair_correct": sum(pp.values()),
                "members_correct": sum(mm.values()),
            }
            if "egg" in rows[0]:
                egg = [r for r in rows if r["egg"]]
                ep, _ = witness_scores(egg)
                entry.update(egg_pairs=len(ep), egg_pair_correct=sum(ep.values()))
            if key == "witness":
                ip, im = witness_scores(rows, "immediate_pred")
                entry.update(
                    immediate_pairs=sum(ip.values()),
                    immediate_members=sum(im.values()),
                    paired_test_pairs=mcnemar(pp, ip),
                    paired_test_members=mcnemar(mm, im),
                )
            result["witness"][key] = entry
    result["target_combined"] = {}
    for policy in ("base", "a0", "a1", "a2", "p", "s2", "s3", "ans"):
        rows = (
            data["target/" + policy + "/sega"]["rows"]
            + data["target/" + policy + "/segb"]["rows"]
        )
        result["target_combined"][policy] = episode_summary(rows)
    return result


def verify_report(result):
    ref = json.loads((ROOT / "data/reference/epact.json").read_text(encoding="utf-8"))
    checked = 0

    def check(actual, expected, label):
        nonlocal checked
        if actual != expected:
            raise ValueError(f"{label}: recomputed {actual!r}, expected {expected!r}")
        checked += 1

    for policy, segments in ref["rl_arc"]["policies"].items():
        for segment, r in segments.items():
            got = (
                result["target_combined"][policy]
                if segment == "combined"
                else result["episodes"]["target/" + policy + "/" + segment]
            )
            for k, v in r.items():
                check(got[k], v, policy + "/" + segment + "/" + k)
    for policy, key in [
        ("base_reasoned", "base_reasoned"),
        ("a0_reasoned", "a0_reasoned"),
    ]:
        got = result["episodes"]["step0/" + policy]
        r = ref["step0"][key]
        check(got["queried_container_opened_eps"], r["opened"], key + "/opened")
        check(got["scored_correct"], r["correct"], key + "/correct")
        check(got["invalid_terminations"], r["invalid"], key + "/invalid")
    check(
        result["episodes"]["step0/a0_immediate"]["queried_container_opened_eps"],
        1,
        "step0/immediate",
    )
    for key, field in [
        ("sft_pairs", "pair_success"),
        ("sft_opened", "queried_container_opened_eps"),
    ]:
        check(
            [result["target_combined"][p][field] for p in ("a0", "s2", "s3")],
            ref["replication"][key],
            key,
        )
    ans = result["target_combined"]["ans"]
    check(ans["pair_success"], 1, "ANS pairs")
    check(ans["answer_counts"]["no"], 101, "ANS no responses")
    check(ans["queried_container_opened_eps"], 10, "ANS opened")
    w = result["witness"]["witness"]
    wr = ref["failure_analysis"]["witness_pairs"]
    check(w["pair_correct"], wr["reasoned"], "witness reason")
    check(w["immediate_pairs"], wr["immediate"], "witness immediate")
    check(
        round(w["paired_test_pairs"]["p_exact_two_sided"], 2),
        wr["reported_p"],
        "witness p",
    )
    check(w["members_correct"], 119, "witness members")
    check(w["immediate_members"], 116, "immediate members")
    for kind, key in [("d", "distill_pairs"), ("s", "label_sft_pairs")]:
        check(
            [
                result["witness"][f"egg/e1_{kind}_s{s}"]["egg_pair_correct"]
                for s in (20261013, 20261014, 20261015)
            ],
            ref["failure_analysis"]["secondary_egg_training"][key],
            key,
        )
    check(result["witness"]["egg/e1_b_r"]["egg_pair_correct"], 5, "egg baseline")
    check(result["witness"]["egg_note"]["egg_pair_correct"], 10, "egg note")
    check(
        result["episodes"]["probe/toilet"]["queried_container_opened_eps"],
        0,
        "toilet fact",
    )
    check(
        result["episodes"]["probe/toilet_directive"]["queried_container_opened_eps"],
        4,
        "toilet directive",
    )
    check(
        result["episodes"]["probe/toilet_directive"]["correct_after_opening"],
        1,
        "toilet reading",
    )
    for key, v in ref["post_k40"].items():
        if key != "source_sha256":
            check(result["post"]["checkup_reasoned_k40"][key], v, "POST K40/" + key)
    for cell, refcell in json.loads(
        (ROOT / "data/reference/post.json").read_text(encoding="utf-8")
    )["cells"].items():
        for key in ("pair_correct", "correct_choices", "sufficient_evidence_episodes"):
            if refcell[key] is not None:
                check(
                    result["post"][cell][key], refcell[key], "POST/" + cell + "/" + key
                )
    return checked


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = recompute()
    result["verified_report_values"] = verify_report(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verified_report_values": result["verified_report_values"],
                "target_pair_scores": {
                    k: v["pair_success"] for k, v in result["target_combined"].items()
                },
                "post_pair_scores": {
                    k: v["pair_correct"] for k, v in result["post"].items()
                },
            }
        )
    )
