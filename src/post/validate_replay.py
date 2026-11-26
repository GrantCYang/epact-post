"""EXT3-G0 (POST_VERDICT.md EXTENSION 3): sim-only replay-fidelity gate.
Replays every member's frozen oracle_script through the live server and
requires the evidence classifier to entail the label on >= 20/24
members before any model call."""

import json
import sys
import argparse
from research_common.paths import ROOT
from pathlib import Path
from post.client import Client
from post.report import evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--pairs", type=Path, default=ROOT / "configs/post/frozen/pairs_post.jsonl"
    )
    args = parser.parse_args()
    port = args.port
    pairs = [json.loads(l) for l in open(args.pairs)]
    cl = Client(port)
    out, ok_n = ([], 0)
    for pair in pairs:
        for mem in pair["members"]:
            cl.call(
                {"cmd": "start", "pair_id": pair["pair_id"], "member": mem["member"]}
            )
            actions = []
            last_pick = None
            for action in mem["oracle_script"]:
                r = cl.call({"cmd": "act", "action": action})
                log = r["privileged"]["log"]
                actions.append({"action": action, "log": log})
                if "try_lift" in action and log.get("ok"):
                    last_pick = action["try_lift"]
                if "put_down" in action and log.get("ok") and last_pick:
                    cl.call({"cmd": "restore", "name": last_pick})
                    last_pick = None
            ev = evidence(actions, pair["candidates"])
            good = ev["entailed"] == mem["label_choice"]
            ok_n += good
            out.append(
                {
                    "ep": f"{pair['pair_id']}_{mem['member']}",
                    "entailed": ev["entailed"],
                    "label": mem["label_choice"],
                    "ok": good,
                }
            )
            print(
                f"[g0] {pair['pair_id']}/{mem['member']}: entailed={ev['entailed']} label={mem['label_choice']} {('OK' if good else 'MISS')}",
                flush=True,
            )
    res = {
        "gate": "EXT3-G0",
        "entailed_ok": ok_n,
        "n": len(out),
        "pass": ok_n >= 20,
        "records": out,
    }
    output = Path("runs/post/validation/ext3_g0.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[g0] {ok_n}/24 -> {('PASS' if res['pass'] else 'FAIL')}", flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
