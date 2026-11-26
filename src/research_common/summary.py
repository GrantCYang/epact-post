"""Print completed-study aggregates without loading a model or simulator."""

import json
from pathlib import Path

result = json.loads(
    (Path(__file__).resolve().parents[2] / "data/reference/epact.json").read_text(
        encoding="utf-8"
    )
)
print("EPACT completed through", result["completed_through"])
for name, arms in result["rl_arc"]["policies"].items():
    r = arms["combined"]
    print(
        name,
        str(r["pair_success"]) + "/" + str(r["pairs"]),
        "opened",
        str(r["queried_container_opened_eps"]) + "/" + str(r["episodes"]),
    )
print(
    "No-Open SFT replication:",
    result["replication"]["sft_pairs"],
    "/55; base",
    result["replication"]["base_pairs"],
)
print("Step 0 reason-first opening: base 22/40, A0 27/40")
print(
    "POST K40:",
    result["post_k40"]["pair_correct"],
    "/12; sufficient evidence",
    result["post_k40"]["sufficient_evidence_episodes"],
    "/24",
)
print("Failure analysis:", json.dumps(result["failure_analysis"], ensure_ascii=False))
