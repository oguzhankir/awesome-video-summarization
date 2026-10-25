# Coverage methodology and research gaps

[Home](../README.md) · [Paper catalog](generated/papers.md) · [Implementation sources](generated/resources.md) · [Latest audit](audits/2026-10-25-weekly.md) · [Overhaul audit](audits/2026-09-08-overhaul.md)

**Literature/source review: 2026-10-25.** This is a selective, evidence-backed handbook, not a systematic review with a completed census of every proceedings volume. A registry can have complete fields while experiment details remain explicitly unverified. Neither a verification date nor a reachable repository establishes successful ML reproduction.

## 1. Inclusion and verification

A core method must solve a defined summarization target; an adjacent resource must explain what it contributes and what target it lacks. Verify title, authors, venue/year and release against the primary publication. Follow paper/project links to author code, then inspect implementation and artifacts separately. A method with multiple variants counts as one publication and multiple result records.

Every new number comes from an identified primary table/page. Exact split identities, boundary files, reference construction and checkpoint selection are recorded or explicitly left unknown. In this release unknown split identities isolate results; no compatible global best result is established. There are no independently reproduced model scores.

Inherited reconstruction cards retain their earlier 2026-09-06 technical-audit dates unless a particular entry was rechecked. The 2026-09-08 overhaul records also retain their field-level dates unless this weekly audit changed them. Exact legacy archive durations and checksums are labeled inherited and were not recomputed from downloaded bytes. No model score was independently reproduced.

## 2. Executed search matrix

The current [2026-10-25 audit](audits/2026-10-25-weekly.md) records a targeted 2026-10-18–25 web/primary-source search, two earlier-paper backfills, and substantial corrections to inherited identities and release descriptions. It adds RVLSM, Semantic-Guided Unsupervised Video Summarization and SoccerHigh, verifies public LVSum/KnowVis source, and does not claim an exhaustive arXiv API census. The matrix and query log below describe earlier passes, with known errors explicitly corrected.

The previous passes recorded 2026-08-09 through 2026-09-19 (11,568 records across `cs.CV`, `cs.MM`, `cs.CL`, `cs.AI` and `cs.HC`) and 2026-09-20 through 2026-10-01 ([2026-10-01 audit](audits/2026-10-01-weekly.md)). The 2026-10-18 maintenance entry reported a **2026-10-01 through 2026-10-18** five-category scan of 3,907 preprints plus targeted earlier-paper follow-up. That count is an inherited search-log claim, not a census reproduced by the current audit. Several admitted papers actually first appeared before October 2026; the corrected identities below distinguish literature backfill from in-window publication. The source matrix remains selective rather than an exhaustive proceedings census.

| Venue/source family | Work performed | Limits |
|---|---|---|
| CVPR, ICCV, ECCV / CVF and ECVA | Landmark supervised/reconstruction methods; full primary-source/repo verification of CSTA, CSUM, SELF-VS, A2Summ, LfVS, LLMVS, SummDiff, CoE, TRINITY and task-specific sources | Selected papers and citation trails; TRINITY's author acceptance claim lacks an independently indexed proceedings entry |
| NeurIPS | seqDPP, Mr. HiSum, QVHighlights and DPP/representation-learning foundations; targeted summarization queries | No complete year-by-year summarization census |
| AAAI | Cycle-SUM, CSNet, DR-DSN and V2Xum-LLM primary sources and pinned V2Xum-LLM repository | Selected primary records |
| ICLR | Exact TripleSumm identity, main/appendix protocols, official artifacts and pinned implementation | Focused 2026 check |
| ACM Multimedia and associated workshops | ACGAN, stepwise adversarial methods, SD-VSum, and follow-up multi-modal script-driven SD-MVSum | Publisher restrictions on some pages; accepted manuscripts and Zenodo releases used |
| TIP, TCSVT, TMM, Pattern Recognition, Machine Vision and Applications | DSNet, AC-SUM-GAN, VideoXum, CAAN, CARLSVS and sparse-reconstruction discovery | Paper-level coverage; CARLSVS is a 2026 Springer publication, not the previously claimed 2025 JVCIR paper |
| TPAMI, WACV, BMVC, ICIP, TALLIP, JCDL | SSPVS, personalized day-long egocentric summaries, causal streaming, SAFCN (TALLIP 2026), JCDL 2026 archival summarization and M2UR metadata | Restricted full text and incomplete venue coverage remain explicit; SAFCN and JCDL bibliographic identities were corrected |
| ACL, EACL, EMNLP | SummScreen, Multi-VidSum, KnowVis and completed QEVA (Findings of EMNLP 2025) primary-source and metric audit | KnowVis proceedings entry is not independently indexed; textual metrics remain separate from keyshot overlap |
| arXiv and author projects | Prior date-window scan plus backfill of SD-MVSum, TRIM, TRIMMER, VASTSum, LVSum, TR-EduVSum, ClipSum, Multi-Video Positional Bias and HAS; October entries include PIGEONS, FramePack-U and AdaFrame | Discovery date is not publication date; corrected primary identifiers below replace erroneous October links |
| Crossref, Wiley, Springer, IEEE, Elsevier and ACM | In-window and 2025–2026 DOI/publisher discovery for query-focused, archival, generic and sports summarizers | Publisher security gates and first-online/issue-date differences are recorded explicitly |
| GitHub and artifact owners | Official/community attribution, pinned source commits (`CSTA`, `A2Summ`, `V2Xum-LLM`, `SummDiff`, `SUM-SR`, `SD-MVSum`, `FramePack-U`, `TRIM`, `ClipSum`, `QEVA`, `Prompts to Summaries`, `Context-Pseudo-Label`), features and checkpoints | Source inspection only; no upstream model execution or large binary download |

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
video summarization submitted 2026-08-09 TO 2026-09-19 cs.CV cs.MM cs.CL cs.AI cs.HC
video highlight detection audio sports 2026; video summary visual narrative agentic 2026
video summarization personalized multi-video causal streaming online 2026
all:video AND submittedDate:[202609200000 TO 202610012359] AND (cat:cs.CV OR cat:cs.MM OR cat:cs.CL OR cat:cs.AI OR cat:cs.HC)
video summarization site:doi.org 2026 CLIP BLIP-2 hierarchical temporal soccer
all:(video AND (summarization OR summary OR keyframe OR keyshot OR highlight)) AND submittedDate:[202610010000 TO 202610182359]
query-focused video summarization 2025 2026 site:doi.org
```

The deliberately broad DSNet query initially suggested AAAI; the verified archival venue is TIP. Search terms are hypotheses, not evidence. GitHub lookups included exact title/author/identifier searches for every admitted recent paper plus historical source leads. [The 2026-09-08 source ledger](audits/2026-09-08-sources.md) preserves overhaul provenance; [the 2026-09-19 audit](audits/2026-09-19-weekly.md) and [2026-10-01 audit](audits/2026-10-01-weekly.md) record earlier incremental searches, and Section 3 below records the 2026-10-18 dispositions.

## 3. Candidates and disposition

| Candidate | Decision and reason |
|---|---|
| Triple-Sum / Triple Sum / Triplet-Sum | Resolved to [TripleSumm](https://arxiv.org/abs/2603.01169); no separate paper established for spelling variants |
| SummDiff | Included as supervised/behavior-supervised diffusion; pinned official repo (`1d4b8ae7`) ; the earlier rank-correlation correction claim was not substantiated by the pinned README and is withdrawn |
| Context-Aware Pseudo-Label Scoring | Included with label-assisted rubric-calibration caveat and pinned commit (`03ba6972`); headline comparison numbers withheld because protocol/citation issues remain |
| CoE / Cut to the Chase | Included as text summarization; training-free with five training-set style examples |
| EdgeVidSum | Included as demo/report and personalized playback; not established as a causal streaming benchmark |
| [SSPVS, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Li_Progressive_Video_Summarization_via_Multimodal_Self-Supervised_Learning_WACV_2023_paper.html) | Included after paper/source audit; self-supervision is pretraining and released epoch selection is test-influenced |
| [Multi-VidSum, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.457/) | Included with dataset, official code and separate evaluator; keyframe-caption alignment metrics stay isolated |
| [CLIP-It!, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/7503cfacd12053d309b6bed5c89de212-Abstract.html) | Included; official repository is a stub and the community source is only a forward-pass sketch |
| [QEVA, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1340/) | Included in 2026-10-18; corrected arXiv mirror is [2604.24052v1](https://arxiv.org/abs/2604.24052v1). MLVU(VS)-Eval has 800 model-generated summaries from 200 videos. Canonical [author repository](https://github.com/wo0jun/QEVA/tree/657cad6bff56e3cb25bf41610c020ce68787abd3) remains a README-only stub |
| [SD-MVSum, arXiv 2025; inspected v2](https://arxiv.org/abs/2510.05652v2) | Script-conditioned multimodal summarization uses script-video and script-transcript weighted cross-modal attention; official code is [IDT-ITI/SD-MVSum](https://github.com/IDT-ITI/SD-MVSum). Earlier October arXiv and commit identifiers in this ledger were erroneous |
| [TRIM, ICASSP 2026](https://arxiv.org/abs/2506.20588v3) | Self-supervised pretraining followed by unsupervised temporal-information learning with a CNN/MLP scorer; not training-free VLM scoring. [Official commit `86a6be3e`](https://github.com/pmishra-ai/TRIM/tree/86a6be3e5b16e7c5116a22a21ed9d4fecd12e6ab) includes source, split manifests and checkpoints; the earlier stub claim was incorrect |
| [TRIMMER, arXiv 2026](https://arxiv.org/abs/2605.01659v1) | Self-supervised representation learning followed by information-theoretic reinforcement learning; not VLM pseudo-summary distillation |
| [VASTSum, IJCNN 2026](https://arxiv.org/abs/2605.09507v1) | Supervised variational importance scoring and decoder-aligned regularization; not multimodal alignment-consistency pretraining |
| [LVSum, arXiv 2026](https://arxiv.org/abs/2604.10024v2) | Timestamp-aware long-video textual-summary benchmark across 13 categories; the earlier scene-importance description and October identifier did not describe this paper |
| [TR-EduVSum, arXiv 2026](https://arxiv.org/abs/2604.07553v1) | Turkish educational dataset with 82 videos and 3,281 independent human summaries; AutoMUP extracts consensus meaning units. It is not a bilingual LLM-synthesized keyshot benchmark |
| [ClipSum, ICPR 2026](https://arxiv.org/abs/2605.11959v1) | Supervised frozen-CLIP/temporal-Transformer/BART abstractive instructional summarization. [Official commit `c3746c1b`](https://github.com/aqeeelmirza/clipsum/tree/c3746c1b02b9e6473136f09735e9aa32b1df3e5d) contains feature extraction, training and evaluation source; the earlier placeholder claim was incorrect |
| [Multi-Video Positional Bias, arXiv 2026](https://arxiv.org/abs/2606.04596v1) | Evaluates Coverage, Directional Positional Bias (DPB) and Middle-Edge Gap (MEG) across input positions; earlier `PC`/`PBI` claims were incorrect |
| [HAS, arXiv 2026](https://arxiv.org/abs/2607.17994v1) | Highlight-guided attention steering preserves continuous video context; supervised steering calibration must be distinguished from training-free inference |
| [FramePack-U, arXiv 2026](https://arxiv.org/abs/2610.16118v1), [AdaFrame, arXiv 2026](https://arxiv.org/abs/2610.19127v1) | Training-free fixed-budget frame packing and advertisement-video deduplication respectively; downstream QA/classification metrics remain separate from keyshot F1 |
| [PIGEONS, arXiv 2026](https://arxiv.org/abs/2610.16066v1) | Adjacent long-video captioning with a persistent identity gallery and evolving narrative memory; uses supervised fine-tuning and reinforcement learning, not unsupervised pseudo-triplets |
| [SAFCN, TALLIP 2026](https://doi.org/10.1145/3856799), [CARLSVS, Machine Vision and Applications 2026](https://doi.org/10.1007/s00138-026-01932-8), [JCDL 2026 archival summarization](https://doi.org/10.1145/3805696.3846037) | Corrected publisher identities for query-focused, sports and archival settings; restricted full-text protocol details remain explicitly unverified |
| [CSTA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Son_CSTA_CNN-based_Spatiotemporal_Attention_for_Video_Summarization_CVPR_2024_paper.html), [CSUM, ICCV 2023](https://arxiv.org/abs/2301.05213), [SELF-VS, CVPRW 2023](https://arxiv.org/abs/2303.15993) | **Resolved in 2026-10-18:** replaced all `"Not independently verified"` protocol fields via full primary-source PDF and repository inspection; added CSTA's pinned code (`b070f7d9`) and four Table 2 rank-correlation rows |
| [Personalized day-long egocentric summaries, TPAMI](https://doi.org/10.1109/TPAMI.2021.3118077) | Included with protocol-separated long-form/RL/personalization caveats and pinned author code |
| [Diversity Promoting Online Sampling, ICIP 2016](https://doi.org/10.1109/ICIP.2016.7532976) | Included as genuinely causal streaming; altered-video/oracle-budget metric remains isolated |
| [M2UR, ICIP 2026](https://doi.org/10.1109/ICIP61757.2026.11630213) | Metadata-only record; PDF security challenge left all unavailable protocol fields unknown |
| [HABSS](https://doi.org/10.1007/s44443-026-01222-3) | Deferred as adjacent preprocessing because output is a candidate-frame pool, not a final summary |
| [Hybrid Spatio-Temporal Feature Representation](https://doi.org/10.21203/rs.3.rs-9730635/v1) | Deferred low-evidence preprint: no dataset-specific final result, budget/solver/reference protocol or code |
| Context Awareness (arXiv 2404.04564) | Thesis-level learning resource candidate; no author code established |
| MF2Summ, MiLoRA-ViSum, SpiVG, compressed-domain graph candidates | Discovered but not admitted as verified method records in this pass |
| Generic video QA, temporal grounding and captioning | Adjacent resources only unless an explicit summarization target is established |
| Medium introductory GAN tutorial | Excluded from recommended lessons because its explanation conflates unsupervised discriminator training with human reference-summary supervision |

## 4. Access and reproducibility gaps

Selected publisher/CVF/author pages rejected automated requests or lacked usable HTML; primary PDFs, arXiv versions or author source were used where available. M2UR (IEEE), SAFCN (ACM TALLIP 2026), JCDL 2026 (ACM) and CARLSVS (Springer 2026) retain explicit full-text access boundaries; the earlier ICME/JVCIR/JCDL-2025 attribution was incorrect. The old QFVS project could not be retrieved. Legacy VASNet Box artifacts return 404 and remain documented as [known link issues](../.github/known-link-issues.json); they are rechecked, not silently ignored. The 2026-10-25 full link attempt returned `retry` for all 495 discovered URLs because the runner could not resolve external hosts. It does not establish target availability or absence. The earlier 2026-09-19 run had the same environment-level limitation for 370 URLs. The differentiated 2026-09-08 snapshot remains the latest useful network observation. Access restrictions, unresolved hosts and missing artifacts are different statuses.

No GPU environments were rebuilt. No datasets, checkpoints or feature archives were downloaded or rehashed. Modern repositories may need large models, a model-serving stack or API access. Dependency and evaluator source were inspected, which supports concrete caveats but not a claim of complete reproduction.

## 5. Next audit priorities

Follow the release status in each paper/resource card rather than assuming every advertised artifact is available. QEVA remains a stub; TRIM and ClipSum have substantive code and need execution and protocol checks. Recheck datasets and execution for the now-public LVSum release; recheck code and anonymous-review artifacts for VASTSum, TRIMMER, TR-EduVSum, Multi-Video Positional Bias, HAS, PIGEONS and AdaFrame. Complete FastPerson/QA-FastPerson and HABSS preprocessing audits, and obtain full texts for publisher-gated SAFCN, CARLSVS, JCDL 2026 and M2UR records. Expand medical/procedural and preference-learning coverage. Resolve TRINITY and SGWIB source conflicts and TripleSumm's appendix number/captioner inconsistencies. Reproduce a controlled supervised baseline and a reward-based baseline using one public split manifest, fixed checkpoint rule and fully specified decoder. Confirm or replace unavailable artifacts without erasing their provenance.

On each meaningful run, update this ledger or add a dated linked audit, update the changelog, preserve old verification dates for untouched claims, and regenerate catalogs. Record exclusions and negative search outcomes as well as additions.
