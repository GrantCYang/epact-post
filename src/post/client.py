"""Run frozen POST evidence-access conditions.

The pair file supplies the task; README.md defines the condition contrasts.
Run identities prevent mixing models or protocols. Failed episodes are retryable.
"""

from __future__ import annotations
import argparse
import base64
import json
import socket
import time
from pathlib import Path
from research_common.run_state import (
    sha256,
    model_identity,
    prepare_run,
    atomic_json,
    reusable_episode,
    completion,
)

FORCE = 'Reply with JSON only: {"deliver": "<object name>"}'
NOTE_HEADER = "[YOUR NOTE — the only thing you keep; rewrite it any turn]"
WINDOW_HEADER = (
    "[RECENT OBSERVATIONS — older ones are GONE; keep what matters in your NOTE]"
)
CURRENT_VIEW = "(The image is your camera's CURRENT view.)"
CHOICE_INTRO = "You are operating a NEW BODY. Its properties may differ from any body you have operated before. Your testing phase is OVER; do not request any more tests."
ACT_PROMPT = 'Reply with optional NOTE lines, then ONE action JSON object on its own line: {"rotate": "left"} or {"rotate": "right"}, {"move": "forward"}, {"try_lift": "<name>"}, {"put_down": true}, or {"done": true}.'
WATCH_PROMPT = "Reply with NOTE lines only (each starting with NOTE:). Your tests are being chosen for you."
FLUSH_LINE = "(all earlier observations are GONE; only your NOTE above remains)"
REASONED_TAIL = 'Think it through step by step first. Then end your reply with the JSON on its own line: {"deliver": "<object name>"}'
K_WINDOW = 4
K_TURNS = 12
BODYCHECK_CELLS = ("checkup", "checkup_flush", "checkup_reasoned")


class Client:
    def __init__(self, port: int):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=900)
        self.f = self.s.makefile("rwb")

    def close(self):
        self.f.close()
        self.s.close()

    def call(self, msg: dict) -> dict:
        self.f.write((json.dumps(msg) + "\n").encode())
        self.f.flush()
        r = json.loads(self.f.readline())
        if not r.get("ok"):
            raise RuntimeError(f"server error: {r.get('error')}")
        return r


def notes_from(reply: str, prev: list) -> list:
    got = [
        l.strip() for l in reply.splitlines() if l.strip().upper().startswith("NOTE:")
    ]
    return got[:6] if got else prev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument(
        "--cell",
        required=True,
        choices=(
            "told_profile",
            "told_profile_reasoned",
            "b0",
            "checkup",
            "checkup_flush",
            "checkup_watch",
            "checkup_reasoned",
            "checkup_watch_reasoned",
        ),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget-s", type=float, default=10800.0)
    ap.add_argument("--k-turns", type=int, default=K_TURNS)
    a = ap.parse_args()
    if a.k_turns < 1 or a.budget_s <= 0:
        ap.error("k-turns and budget-s must be positive")
    pairs = [
        json.loads(l)
        for l in a.pairs.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    expected_ids = [
        f"{p['pair_id']}_{m['member']}" for p in pairs for m in p["members"]
    ]
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Empty or duplicate POST task identities")
    source_names = ("client.py", "body.py", "server.py", "score.py", "report.py")
    identity = {
        "schema_version": 1,
        "cell": a.cell,
        "k_turns": a.k_turns,
        "pairs_sha256": sha256(a.pairs),
        "model_files": model_identity(a.model_path),
        "sources": {n: sha256(Path(__file__).with_name(n)) for n in source_names},
        "decoding": {
            "do_sample": False,
            "checkup_max_new_tokens": 160,
            "immediate_choice_max_new_tokens": 96,
            "reasoned_choice_max_new_tokens": 512,
        },
    }
    prepare_run(a.out, identity)
    (a.out / "episodes").mkdir(exist_ok=True)
    (a.out / "frames").mkdir(exist_ok=True)
    completion(a.out, expected_ids)
    import sys as _sys
    from post.score import parse_action, parse_choice
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    import io

    t_load = time.monotonic()
    processor = AutoProcessor.from_pretrained(a.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(f"[{a.cell}] model loaded {time.monotonic() - t_load:.0f}s", flush=True)

    def gen(img, text, max_new=160):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text},
                ],
            }
        ]
        inp = processor.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        n = inp["input_ids"].shape[1]
        with torch.inference_mode():
            o = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return processor.batch_decode(o[:, n:], skip_special_tokens=True)[0]

    def img_of(rep, tag, ep):
        raw = base64.b64decode(rep["frame_png_b64"])
        (a.out / "frames" / f"{ep}_{tag}.png").write_bytes(raw)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    cl = Client(a.port)
    t0, done, gen_s = (time.monotonic(), 0, 0.0)
    for pair in pairs:
        names = tuple((c["name"] for c in pair["candidates"]))
        for mem in pair["members"]:
            epid = f"{pair['pair_id']}_{mem['member']}"
            epath = a.out / "episodes" / f"{epid}.json"
            if reusable_episode(epath):
                continue
            if time.monotonic() - t0 > a.budget_s:
                print(f"[{a.cell}] budget reached at {epid}", flush=True)
                break
            ep = {
                "cell": a.cell,
                "pair_id": pair["pair_id"],
                "dim": pair["dim"],
                "member": mem["member"],
                "label_choice": mem["label_choice"],
                "candidates": pair["candidates"],
                "theta_NOT_SHOWN": mem["theta"],
                "turns": [],
                "window": [],
                "note": [],
                "actions": [],
                "choice_reply": None,
                "parsed_choice": None,
                "delivery": None,
            }
            try:
                st = cl.call(
                    {
                        "cmd": "start",
                        "pair_id": pair["pair_id"],
                        "member": mem["member"],
                    }
                )
                img = img_of(st, "t0", epid)
                ep["window"].append(st["odometry"])
                n_turn = 0
                ep["k_turns"] = a.k_turns
                intro = pair["intro"]
                if a.k_turns != K_TURNS:
                    intro = intro.replace(
                        "up to 12 action turns", f"up to {a.k_turns} action turns"
                    )
                if a.cell in BODYCHECK_CELLS:
                    while n_turn < a.k_turns:
                        ctx = "\n".join(
                            [
                                intro,
                                NOTE_HEADER,
                                "\n".join(ep["note"]) or "NOTE: (empty)",
                                WINDOW_HEADER,
                                *ep["window"][-K_WINDOW:],
                                ACT_PROMPT,
                                CURRENT_VIEW,
                            ]
                        )
                        t1 = time.monotonic()
                        raw = gen(img, ctx)
                        gen_s += time.monotonic() - t1
                        ep["note"] = notes_from(raw, ep["note"])
                        act = parse_action(raw)
                        ep["turns"].append(
                            {
                                "phase": "checkup",
                                "model_visible_text": ctx,
                                "raw_reply": raw.strip(),
                                "parsed": act,
                            }
                        )
                        n_turn += 1
                        if act["kind"] == "done":
                            break
                        if act["kind"] == "invalid":
                            ep["window"].append(
                                "[action refused] not a valid action. That turn is spent."
                            )
                            continue
                        r = cl.call({"cmd": "act", "action": act["action"]})
                        ep["actions"].append(
                            {
                                "action": act["action"],
                                "log": r["privileged"]["log"],
                                "agent": r["privileged"]["agent"],
                            }
                        )
                        obs = (
                            "You commanded: "
                            + json.dumps(act["action"])
                            + "\n"
                            + "\n".join(r["lines"] + [r["odometry"]])
                        )
                        ep["window"].append(obs)
                        img = img_of(r, f"t{n_turn}", epid)
                elif a.cell in ("checkup_watch", "checkup_watch_reasoned"):
                    moved_seen = False
                    last_pick = None
                    for i, action in enumerate(mem["oracle_script"]):
                        r = cl.call({"cmd": "act", "action": action})
                        log_i = r["privileged"]["log"]
                        ep["actions"].append(
                            {
                                "action": action,
                                "log": log_i,
                                "agent": r["privileged"]["agent"],
                            }
                        )
                        obs = (
                            "The harness commanded: "
                            + json.dumps(action)
                            + "\n"
                            + "\n".join(r["lines"] + [r["odometry"]])
                        )
                        ep["window"].append(obs)
                        img = img_of(r, f"w{i}", epid)
                        if a.cell == "checkup_watch_reasoned":
                            if "try_lift" in action and log_i.get("ok"):
                                last_pick = action["try_lift"]
                            if "put_down" in action and log_i.get("ok") and last_pick:
                                cl.call({"cmd": "restore", "name": last_pick})
                                last_pick = None
                                rr = cl.call({"cmd": "render"})
                                img = img_of(rr, f"w{i}r", epid)
                        if a.cell == "checkup_watch_reasoned":
                            fire = False
                            for l in r["lines"]:
                                if l.startswith("[lift outcome]"):
                                    fire = True
                                elif l.startswith("[move outcome]"):
                                    if "moved forward" in l:
                                        fire = fire or not moved_seen
                                        moved_seen = True
                                    else:
                                        fire = True
                        else:
                            fire = any(
                                (
                                    l.startswith("[move outcome]")
                                    or l.startswith("[lift outcome]")
                                    for l in r["lines"]
                                )
                            )
                        if fire:
                            ctx = "\n".join(
                                [
                                    pair["intro"],
                                    NOTE_HEADER,
                                    "\n".join(ep["note"]) or "NOTE: (empty)",
                                    WINDOW_HEADER,
                                    *ep["window"][-K_WINDOW:],
                                    WATCH_PROMPT,
                                    CURRENT_VIEW,
                                ]
                            )
                            t1 = time.monotonic()
                            raw = gen(img, ctx)
                            gen_s += time.monotonic() - t1
                            ep["note"] = notes_from(raw, ep["note"])
                            ep["turns"].append(
                                {
                                    "phase": "watch",
                                    "model_visible_text": ctx,
                                    "raw_reply": raw.strip(),
                                }
                            )
                told = ""
                if a.cell in ("told_profile", "told_profile_reasoned"):
                    told = mem["profile"] + "\n" + pair["facts_distances"] + "\n"
                elif a.cell == "b0":
                    told = pair["facts_distances"] + "\n"
                core = pair["choice_core"]
                max_new_choice = 96
                if a.cell in (
                    "told_profile_reasoned",
                    "checkup_reasoned",
                    "checkup_watch_reasoned",
                ):
                    core = core.rsplit("\n", 1)[0] + "\n" + REASONED_TAIL
                    max_new_choice = 512
                win = (
                    [FLUSH_LINE]
                    if a.cell == "checkup_flush"
                    else ep["window"][-K_WINDOW:]
                )
                ctx = "\n".join(
                    [
                        CHOICE_INTRO,
                        NOTE_HEADER,
                        "\n".join(ep["note"]) or "NOTE: (empty)",
                        WINDOW_HEADER,
                        *win,
                        told + core,
                        CURRENT_VIEW,
                    ]
                )
                t1 = time.monotonic()
                raw = gen(img, ctx, max_new=max_new_choice)
                gen_s += time.monotonic() - t1
                ep["choice_reply"] = raw.strip()
                ep["turns"].append(
                    {
                        "phase": "choice",
                        "model_visible_text": ctx,
                        "raw_reply": raw.strip(),
                    }
                )
                ep["parsed_choice"] = parse_choice(raw, names)
                chosen = ep["parsed_choice"]["choice"]
                ep["choice_correct"] = chosen == mem["label_choice"] if chosen else None
                if chosen:
                    ep["delivery"] = cl.call({"cmd": "deliver", "name": chosen})
                    ep["delivery"].pop("ok", None)
            except Exception as e:
                ep["ERROR"] = repr(e)[:400]
            atomic_json(epath, ep)
            done += 1
            print(
                f"[{a.cell}] {epid} acts={len(ep['actions'])} choice={(ep.get('parsed_choice') or {}).get('choice')} label={mem['label_choice']} correct={ep.get('choice_correct')}",
                flush=True,
            )
        else:
            continue
        break
    rep = {
        "cell": a.cell,
        "wall_s": round(time.monotonic() - t0, 1),
        "generation_s": round(gen_s, 1),
        "episodes_run": done,
        **completion(a.out, expected_ids),
    }
    atomic_json(a.out / "report.json", rep)
    cl.close()
    print(
        f"[{a.cell}] done {done} episodes, wall {rep['wall_s']}s, gen {rep['generation_s']}s",
        flush=True,
    )
    return 0 if rep["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
