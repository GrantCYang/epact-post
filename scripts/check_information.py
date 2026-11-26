"""Check each episode's prefix evidence against independent recorded references."""

import json
from pathlib import Path
from collections import Counter
from post.evidence_rule import evidence

ROOT = Path(__file__).resolve().parents[1]
datasets = json.loads((ROOT / "data/records/records.json").read_text(encoding="utf-8"))[
    "datasets"
]
refs = json.loads(
    (ROOT / "data/post-figures/information-reference.json").read_text(encoding="utf-8")
)["episodes"]
totals = Counter()
for ref in refs:
    rows = datasets[ref["condition"]]["rows"]
    matches = [
        r
        for r in rows
        if (r["pair_id"], r["member"]) == (ref["pair_id"], ref["member"])
    ]
    if len(matches) != 1:
        raise ValueError("Missing/duplicate POST member")
    r = matches[0]
    acts = r["actions"]
    prev = {}
    changes = 0
    for index in range(1, len(acts) + 1):
        state = evidence(acts[:index], r["candidates"])
        changes += state["feas"] != prev
        prev = state["feas"]
    if (
        changes != ref["changed_turns"]
        or evidence(acts, r["candidates"]) != ref["final_evidence"]
    ):
        raise ValueError("POST prefix evidence changed")
    if (
        len(acts) != ref["executed_actions"]
        or r["turn_kinds"].count("invalid") != ref["invalid_turns"]
    ):
        raise ValueError("POST action coverage changed")
    totals[ref["condition"]] += changes
if list(totals.values()) != [20, 19, 72] or len(refs) != 72:
    raise ValueError("POST cohort counts changed")
print(
    json.dumps(
        {"prefix_replayed_episodes": len(refs), "feasibility_changes": dict(totals)}
    )
)
