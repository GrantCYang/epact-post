# Scoring and figure inputs

`records/` holds 1,644 compact records. Task IDs, pairing, categorical answers, correctness, successful opening calls and POST action/outcome fields are preserved. Original model replies, source paths and provenance blobs are omitted. POST choices are the frozen parser outputs; the public replay checks these choices rather than claiming to reparse removed replies.

`reference/` holds independent reported totals. The prefix replay compares all 72 checkup members with independently saved feasibility-event references, not only the combined totals. Invalid turns remain counted and cannot create evidence. The standalone rule is unchanged from the recorded analysis.

`epact-figures/` holds two native witness renders, crop coordinates, categorical answers, the class note and aggregate target strata/light-use tables. `post-figures/` holds the worked pair, catalogue coordinates, three displayed action timelines and prefix-evidence references. The scripts read these files only. The pair, handoff and light figures in `paper/figures/` are finished figures from the manuscript source, not rebuilt by the CPU check.

`benchmark/` contains the exact 72-pair development selection and the 55-pair target selection (32 reused development pairs and 23 new pairs), with original scene modifications, house indices, poses and oracle plans. Its manifest pins the preserved ProcTHOR validation input and simulator build. The 352 original start views are preserved in two separate checksummed ZIP archives, outside the compact arXiv bundles; the restore command validates membership, byte hashes and paths before installation. The full training/episode attachment stays private. Newly generated inputs are not represented as identical to the originals. Frozen POST specifications and oracle scripts are included separately in `configs/post/frozen/`.

Original numeric tables are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), attributed to Yixuan Yang and the relevant paper. Simulator imagery retains its upstream terms and is not relicensed as code.
