# EPACT and POST

EPACT separates acquiring evidence from using it. Supplied evidence yields **44–57/72** correct pairs, while autonomous interaction yields **7–8/72**. POST asks whether planners can discover body limits: a stated profile with reasoning yields **7/12** correct pairs, watched checkups **9/12**, and self-run checkups **1/12** or **0/12**.

[EPACT paper](paper/main.pdf) · [POST paper](post/paper/main.pdf) · [Data](data/README.md) · [Downloads](https://github.com/GrantCYang/epact-post/releases/tag/v0.3.1)

The manuscripts are available here as preprints. Neither has yet been posted to arXiv; permanent article links will be added when available.

## CPU check

```sh
python -m pip install -e ".[figures]"
python scripts/replay.py
```

This recomputes 119 report values from 1,644 compact scoring records, replays the unchanged POST evidence rule over every action prefix in 72 checkup episodes, and regenerates the teaser, egg, reach and POST results figures. It checks saved outcomes; it makes no model or simulator calls.

## EPACT

`epact data generate --help` exposes the eight-family pair generator. `epact policy --help` exposes explicit-input serving, evaluation, trajectory SFT and frozen evaluation recipes. The original action-only presets remain under `epact original`. The exact frozen task definitions now ship in `data/benchmark`: 72 development pairs and the 55 target pairs (32 reused development pairs plus 23 new pairs). Original IDs, house indices, scene changes, poses and oracle plans are retained. The full training/episode attachment stays private. The 352 original start PNGs are [separate companion archives](https://github.com/GrantCYang/epact-post/releases/tag/v0.3.1), each under 30 MB, and are not included in the compact arXiv bundles; `start-views.json` binds their bytes and pixels. ProcTHOR houses and the pinned simulator remain upstream dependencies. Newly rendered scenes have not been verified as byte-identical.

```sh
epact data frozen check
epact data frozen restore-images --archives EPACT_Frozen_Start_Views_part01.zip EPACT_Frozen_Start_Views_part02.zip --out restored-start-views
epact data frozen check --images restored-start-views --dataset procthor-10k/val.jsonl.gz
epact data frozen plan --suite development --dataset procthor-10k/val.jsonl.gz --model-path model --output-dir new-evaluation
```

The plan checks the original house-input hash and prints executable server/evaluation argument lists for one reason-first evaluation. It does not claim to reproduce historical training or all policy conditions; those recipes remain separately listed in `configs/epact/reason_runs.json`.

The egg figure uses two original witness PNGs, native pixel crops at recorded coordinates, categorical recorded answers and the exact class note. No full model replies or RL training orchestrator are included.

## POST

`post --help` exposes the body-rule layer, frozen-pair builder, simulator server and run/report commands. Twelve frozen pairs and their checkup scripts are in `configs/post/frozen`. `src/post/evidence_rule.py` is the recorded sufficient-evidence rule, copied unchanged. Its action-prefix replay recovers **20, 19 and 72** feasibility-changing turns and **2, 2 and 24** sufficient-evidence episodes.

The reach example was chosen to show the 40-turn rotation loop, not sampled. No checkup condition solved that pair. Its figure uses logged catalogue coordinates and the exact three action timelines shown.

```bibtex
@misc{yang2026post,
  author = {Yixuan Yang},
  title = {POST: Vision--Language Planners Use Stated Body Limits but Rarely Discover Them},
  year = {2026}
}
```

## Dependencies and limits

Qwen3-VL-2B-Instruct, Qwen3-VL-8B-Instruct and Qwen3-VL-30B-A3B-Instruct are external; POST uses the 8B planner. `configs/model_identity.json` records recovered metadata hashes and the preserved 2B checkout revision. It explicitly distinguishes preserved checkout identity from verified historical run binding; the latter remains incomplete, especially for 30B. No weights ship. AI2-THOR uses the pinned CloudRendering build named in `src/research_common/runtime.py`; ProcTHOR scenes remain external. Model and simulator execution has not been rerun in a relocated environment.

Original code is MIT licensed; original numeric tables are CC BY 4.0. See [third-party terms](THIRD_PARTY.md). Build each preprint from `paper/` or `post/paper/` with PDFLaTeX and BibTeX:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Version **0.3.1** updates the manuscripts, related work and release documentation. All numerical tables, scoring records, frozen tasks and scientific implementations are unchanged. Three cropped figure PDFs have obsolete off-crop text removed; their rendered pixels are unchanged. Each source bundle prepared for arXiv includes its paper's styles, bibliography and figures, plus the shared method code and compact data as ancillary files. The companion start-image archives are separate inputs for checking the original images; they are not required for the documented CPU score and figure replay.
