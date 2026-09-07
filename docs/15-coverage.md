# Coverage methodology and research gaps

[Home](../README.md) · [Paper catalog](generated/papers.md) · [Implementation sources](generated/resources.md) · [Overhaul audit](audits/2026-09-08-overhaul.md)

**Literature/source review: 2026-09-08.** This is a selective, evidence-backed handbook, not a systematic review with a completed census of every proceedings volume. A registry can have complete fields while experiment details remain explicitly unverified. Neither a verification date nor a reachable repository establishes successful ML reproduction.

## 1. Inclusion and verification

A core method must solve a defined summarization target; an adjacent resource must explain what it contributes and what target it lacks. Verify title, authors, venue/year and release against the primary publication. Follow paper/project links to author code, then inspect implementation and artifacts separately. A method with multiple variants counts as one publication and multiple result records.

Every new number comes from an identified primary table/page. Exact split identities, boundary files, reference construction and checkpoint selection are recorded or explicitly left unknown. In this release unknown split identities isolate results; no compatible global best result is established. There are no independently reproduced model scores.

Inherited reconstruction cards retain their earlier 2026-09-06 technical-audit dates unless a particular entry was rechecked. Exact legacy archive durations and checksums are labeled inherited and were not recomputed from downloaded bytes. SummDiff's experiment configuration was rechecked in this revision; its model and numerical results were not reproduced.

## 2. Executed search matrix

Searches used foundations through 2026-09-08, with recent discovery concentrated on 2023–2026. This table states actual depth rather than suggesting every venue was exhaustively searched.

| Venue/source family | Work performed | Limits |
|---|---|---|
| CVPR, ICCV, ECCV / CVF and ECVA | Landmark supervised/reconstruction methods; CSTA, A2Summ, LfVS, LLMVS, SummDiff, CoE and task-specific sources | Selected papers and citation trails, not all accepted-paper lists |
| NeurIPS | seqDPP and DPP/representation-learning foundations; targeted summarization queries | No complete year-by-year summarization census |
| AAAI | Cycle-SUM, CSNet, DR-DSN and V2Xum-LLM primary sources | Selected primary records |
| ICLR | Exact TripleSumm identity, main/appendix protocols, official artifacts | Focused 2026 check |
| ACM Multimedia and associated workshops | ACGAN, stepwise adversarial methods, SD-VSum; author code and project evidence | Publisher restrictions on some pages; accepted manuscripts used |
| TIP, TCSVT, TMM, Pattern Recognition | DSNet, AC-SUM-GAN, VideoXum, CAAN and sparse-reconstruction discovery | Paper-level coverage; no journal-wide census |
| TPAMI, WACV, BMVC, ICIP | Targeted venue/topic discovery, WACV SSPVS and TPAMI personalized egocentric candidates identified | Full benchmark/artifact audit deferred; BMVC/ICIP searches did not establish a complete accepted corpus |
| ACL, EACL, EMNLP | SummScreen and video-to-text derivatives; keyframe-caption and reference-free evaluation candidates | Textual metrics reviewed separately from keyshot overlap |
| arXiv and author projects | Recent training-free, zero/few-shot, long-form, query and multimodal variants | Preprints remain labeled; indexing delay and later revisions possible |
| GitHub and artifact owners | Official/community attribution, pinned source files, features, checkpoint listings and learner resources | Source inspection only; no upstream model execution or large binary download |

Representative root queries actually issued include:

```text
video summarization submodular mixtures Gygli 2015 supervised diverse sequential DPP Gong 2014
video summarization contrastive self supervised learning 2021 2022 2023 official paper
DSNet anchor free anchor based video summarization AAAI 2021 paper
video summarization graph attention PGL SUM VASNet CSTA paper
site.openaccess.thecvf.com video summarization graph 2019 2021 relation
video summarization + Triple-Sum / TripleSumm / Triple Sum / Triplet-Sum
video summarization zero-shot 2025 github
video summarization training-free 2024 2025 2026
video summarization online 2024 2025; video summarization streaming 2025
"video summarization" "TPAMI" "ICIP" 2024 2025
"video summarization" "BMVC" "WACV" self supervised
"video summarization" "EMNLP" "ACL" multimodal
"video summarization" "NeurIPS" "TMM" survey 2025 2026
site.bmvc2021-virtualconference.com video summarization
site.openaccess.thecvf.com WACV2023 self supervised video summarization SSPVS
video summarization ICIP 2024 2025 official
video summarization TPAMI survey 2022 2024
```

The deliberately broad DSNet query initially suggested AAAI; the verified archival venue is TIP. Search terms are hypotheses, not evidence. GitHub lookups included exact paper/author names, Cycle-SUM name variants, VASNet, DSNet, VideoXum, QFVS, DR-DSN, SUM-GAN-AAE and PGL-SUM. [The source ledger](audits/2026-09-08-sources.md) preserves code-search decisions and primary benchmark provenance.

## 3. Candidates and disposition

| Candidate | Decision and reason |
|---|---|
| Triple-Sum / Triple Sum / Triplet-Sum | Resolved to [TripleSumm](https://arxiv.org/abs/2603.01169); no separate paper established for spelling variants |
| SummDiff | Included as supervised/behavior-supervised diffusion; no longer excluded by the former unsupervised-only scope |
| Context-Aware Pseudo-Label Scoring | Included with label-assisted rubric-calibration caveat; headline comparison numbers withheld because protocol/citation issues remain |
| CoE / Cut to the Chase | Included as text summarization; training-free with five training-set style examples |
| EdgeVidSum | Included as demo/report and personalized playback; not established as a causal streaming benchmark |
| [SSPVS, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Li_Progressive_Video_Summarization_via_Multimodal_Self-Supervised_Learning_WACV_2023_paper.html) | Primary identity and [author code](https://github.com/HopLee6/SSPVS-PyTorch) found; full protocol/feature audit remains a next entry |
| [Keyframe-caption pairs, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.457/) | Primary task/citation verified; detailed dataset, losses and reproduction audit deferred |
| [QEVA, EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.1340/) | Primary evaluation contribution found; judge/metric and artifact audit deferred |
| [Personalized day-long egocentric summaries, TPAMI](https://doi.org/10.1109/TPAMI.2021.3118077) | Author manuscript discovered; benchmark/code audit deferred |
| Context Awareness (arXiv 2404.04564) | Thesis-level learning resource candidate; no author code established |
| MF2Summ, MiLoRA-ViSum, SpiVG, compressed-domain graph candidates | Discovered but not admitted as verified method records in this pass |
| Generic video QA, temporal grounding and captioning | Adjacent resources only unless an explicit summarization target is established |
| Medium introductory GAN tutorial | Excluded from recommended lessons because its explanation conflates unsupervised discriminator training with human reference-summary supervision |

## 4. Access and reproducibility gaps

Selected publisher/CVF/author pages rejected automated requests or lacked usable HTML; primary PDFs, arXiv versions or author source were used where available. The old QFVS project could not be retrieved. Legacy VASNet Box artifacts return 404 and remain documented as [known link issues](../.github/known-link-issues.json); they are rechecked every run, not silently ignored. ActivityNet's legacy parent URL has a certificate mismatch and the TVR host failed DNS in the local link pass. Access restrictions, unresolved hosts and missing artifacts are different statuses.

No GPU environments were rebuilt. No datasets, checkpoints or feature archives were downloaded or rehashed. Modern repositories may need large models, a model-serving stack or API access. Dependency and evaluator source were inspected, which supports concrete caveats but not a claim of complete reproduction.

## 5. Next audit priorities

Expand WACV/BMVC/ICIP and TPAMI coverage with paper-level protocol extraction. Complete SSPVS and language-output evaluation candidates. Add stronger causal-streaming, multi-camera, medical/procedural, sports, multilingual and preference-learning coverage. Resolve TripleSumm's appendix number/captioner inconsistencies against an author release. Reproduce a controlled supervised baseline and a reward-based baseline using the same held-out protocol. Confirm or replace unavailable artifacts without erasing their provenance.

On each meaningful run, update this ledger or add a dated linked audit, update the changelog, preserve old verification dates for untouched claims, and regenerate catalogs. Record exclusions and negative search outcomes as well as additions.
