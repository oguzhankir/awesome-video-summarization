# Coverage methodology and research gaps

[Home](../README.md) · [Paper catalog](generated/papers.md) · [Implementation sources](generated/resources.md) · [Latest audit](audits/2026-10-01-weekly.md) · [Overhaul audit](audits/2026-09-08-overhaul.md)

**Literature/source review: 2026-10-18.** This is a selective, evidence-backed handbook, not a systematic review with a completed census of every proceedings volume. A registry can have complete fields while experiment details remain explicitly unverified. Neither a verification date nor a reachable repository establishes successful ML reproduction.

## 1. Inclusion and verification

A core method must solve a defined summarization target; an adjacent resource must explain what it contributes and what target it lacks. Verify title, authors, venue/year and release against the primary publication. Follow paper/project links to author code, then inspect implementation and artifacts separately. A method with multiple variants counts as one publication and multiple result records.

Every new number comes from an identified primary table/page. Exact split identities, boundary files, reference construction and checkpoint selection are recorded or explicitly left unknown. In this release unknown split identities isolate results; no compatible global best result is established. There are no independently reproduced model scores.

Inherited reconstruction cards retain their earlier 2026-09-06 technical-audit dates unless a particular entry was rechecked. The 2026-09-08 overhaul records also retain their field-level dates unless this weekly audit changed them. Exact legacy archive durations and checksums are labeled inherited and were not recomputed from downloaded bytes. No model score was independently reproduced.

## 2. Executed search matrix

The previous passes covered 2026-08-09 through 2026-09-19 (11,568 records across `cs.CV`, `cs.MM`, `cs.CL`, `cs.AI` and `cs.HC`) and 2026-09-20 through 2026-10-01 ([2026-10-01 audit](audits/2026-10-01-weekly.md)). The latest pass covers **2026-10-01 through 2026-10-18** across `cs.CV`, `cs.MM`, `cs.CL`, `cs.AI`, and `cs.HC` (screening 3,907 video/multimodal preprints in that window), plus targeted 2025–2026 publisher/DOI and deferred-item follow-up (`QEVA`, `SAFCN`, `CARLSVS`, `JCDL 2025`, and primary-source verification of `CSTA`, `CSUM`, `SELF-VS`, `SummDiff`, `SUM-SR`, `A2Summ`, `V2Xum-LLM`, `Prompts to Summaries`, and `Context-Aware Pseudo-Label Scoring`). This table states actual depth rather than suggesting every venue was exhaustively searched.

| Venue/source family | Work performed | Limits |
|---|---|---|
| CVPR, ICCV, ECCV / CVF and ECVA | Landmark supervised/reconstruction methods; full primary-source/repo verification of CSTA, CSUM, SELF-VS, A2Summ, LfVS, LLMVS, SummDiff, CoE, TRINITY and task-specific sources | Selected papers and citation trails; TRINITY's author acceptance claim lacks an independently indexed proceedings entry |
| NeurIPS | seqDPP, Mr. HiSum, QVHighlights and DPP/representation-learning foundations; targeted summarization queries | No complete year-by-year summarization census |
| AAAI | Cycle-SUM, CSNet, DR-DSN and V2Xum-LLM primary sources and pinned V2Xum-LLM repository | Selected primary records |
| ICLR | Exact TripleSumm identity, main/appendix protocols, official artifacts and pinned implementation | Focused 2026 check |
| ACM Multimedia and associated workshops | ACGAN, stepwise adversarial methods, SD-VSum, and follow-up multi-modal script-driven SD-MVSum | Publisher restrictions on some pages; accepted manuscripts and Zenodo releases used |
| TIP, TCSVT, TMM, Pattern Recognition, JVCIR | DSNet, AC-SUM-GAN, VideoXum, CAAN, CARLSVS and sparse-reconstruction discovery | Paper-level coverage; CARLSVS Elsevier full text is paywalled |
| TPAMI, WACV, BMVC, ICIP, ICME, JCDL | SSPVS, personalized day-long egocentric summaries, causal streaming, SAFCN (ICME 2025), JCDL 2025 archival summarization and M2UR metadata | SAFCN, JCDL 2025 and M2UR full PDFs are publisher-gated; no complete venue census |
| ACL, EACL, EMNLP | SummScreen, Multi-VidSum, KnowVis and completed QEVA (Findings of EMNLP 2025) primary-source and metric audit | KnowVis proceedings entry is not independently indexed; textual metrics remain separate from keyshot overlap |
| arXiv and author projects | Full five-category date-window scan (`2026-10-01` to `2026-10-18`: SD-MVSum, TRIM, TRIMMER, VASTSum, LVSum, TR-EduVSum, ClipSum, Multi-Video Positional Bias, HAS, PIGEONS, FramePack-U, AdaFrame) | Preprints remain labeled; indexing delay, later revisions and non-arXiv papers require publisher follow-up |
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
| SummDiff | Included as supervised/behavior-supervised diffusion; pinned official repo (`1d4b8ae7`) and documented its updated rank-correlation script note |
| Context-Aware Pseudo-Label Scoring | Included with label-assisted rubric-calibration caveat and pinned commit (`03ba6972`); headline comparison numbers withheld because protocol/citation issues remain |
| CoE / Cut to the Chase | Included as text summarization; training-free with five training-set style examples |
| EdgeVidSum | Included as demo/report and personalized playback; not established as a causal streaming benchmark |
| [SSPVS, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Li_Progressive_Video_Summarization_via_Multimodal_Self-Supervised_Learning_WACV_2023_paper.html) | Included after paper/source audit; self-supervision is pretraining and released epoch selection is test-influenced |
| [Multi-VidSum, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.457/) | Included with dataset, official code and separate evaluator; keyframe-caption alignment metrics stay isolated |
| [CLIP-It!, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/7503cfacd12053d309b6bed5c89de212-Abstract.html) | Included; official repository is a stub and the community source is only a forward-pass sketch |
| [QEVA, EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.1340/) | **Completed in 2026-10-18:** full paper (`2604.24052v1`) and official stub (`657cad6b`) audited; added `qeva` paper, `mlvu-vs-eval` meta-evaluation dataset, `qeva-code-stub` resource, and Table 2 correlation rows |
| [SD-MVSum, arXiv 2025/2026](https://arxiv.org/abs/2510.05652v2) | Included with full paper/code/Zenodo audit (`e88adbf3`), `sm-videoxum` and `sm-mrhisum` dataset cards, and six protocol-isolated Table 1/2 result rows |
| [TRIM, arXiv 2025/2026](https://arxiv.org/abs/2506.20588v3) | Included as training-free VLM/CLIP summarizer with official stub (`86a6be3e`) and six Table 1/2 SumMe/TVSum full-dataset F1 and rank-correlation rows |
| [TRIMMER, arXiv 2026](https://arxiv.org/abs/2605.01659v1) | Included as zero-human-label LLM scene-description + CLIP similarity scorer, with four Table 1 SumMe/TVSum rank-correlation rows |
| [VASTSum, arXiv 2026](https://arxiv.org/abs/2605.09507v1) | Included as supervised four-stream (Vision, Audio, Subtitle, Transcript) fusion with eight Table 1 TVT/5FCV SumMe/TVSum rank-correlation rows |
| [LVSum, arXiv 2026](https://arxiv.org/abs/2604.10024v2) | Included with long-form (10–55 min) `lvsum` dataset card and four Table 4 1-fps MLLM rank-correlation rows |
| [TR-EduVSum, arXiv 2026](https://arxiv.org/abs/2604.07553v1) | Included with Turkish computer science lecture `tr-eduvsum` dataset card and AutoMUP multi-reference consensus documentation |
| [ClipSum, arXiv 2026](https://arxiv.org/abs/2605.11959v1) | Included with full paper audit and official placeholder stub (`c3746c1b`) |
| [Multi-Video Positional Bias, arXiv 2026](https://arxiv.org/abs/2606.04596v1) | Included as multi-video MLLM summarization evaluation/bias analysis (`PC` and `PBI` metrics) |
| [HAS, arXiv 2026](https://arxiv.org/abs/2607.17994v1), [FramePack-U, arXiv 2026](https://arxiv.org/abs/2610.16118v1), [AdaFrame, arXiv 2026](https://arxiv.org/abs/2610.19127v1) | Included as training-free keyframe/token selection and deduplication pipelines for MLLM/VLM video understanding and advertisement analysis; downstream QA/classification metrics kept separate from keyshot F1 |
| [PIGEONS, arXiv 2026](https://arxiv.org/abs/2610.16066v1) | Included as unsupervised dual-prompting video-to-text summarization via pseudo-triplets |
| [SAFCN, ACM TOMM 2026](https://doi.org/10.1145/3856799), [CARLSVS, MVA 2026](https://doi.org/10.1007/s00138-026-01932-8), [JCDL 2025 archival summarization](https://doi.org/10.1145/3805696.3846037) | Included as DOI/abstract-verified specialized records for query-focused and historical-archive summarization; paywalled full-text fields explicitly marked |
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

Selected publisher/CVF/author pages rejected automated requests or lacked usable HTML; primary PDFs, arXiv versions or author source were used where available. M2UR's IEEE PDF, SAFCN's ICME 2025 PDF, JCDL 2025's IEEE PDF, and CARLSVS's Elsevier PDF remain publisher-gated and are documented with explicit metadata boundaries. The old QFVS project could not be retrieved. Legacy VASNet Box artifacts return 404 and remain documented as [known link issues](../.github/known-link-issues.json); they are rechecked, not silently ignored. The last full link-audit run, on 2026-09-19, saw environment-level DNS failure for all 370 URLs and classified every endpoint as unresolved; this incremental scan did not repeat that audit. The differentiated 2026-09-08 snapshot remains the latest useful network observation. Access restrictions, unresolved hosts and missing artifacts are different statuses.

No GPU environments were rebuilt. No datasets, checkpoints or feature archives were downloaded or rehashed. Modern repositories may need large models, a model-serving stack or API access. Dependency and evaluator source were inspected, which supports concrete caveats but not a claim of complete reproduction.

## 5. Next audit priorities

Monitor public code and dataset uploads for LVSum, VASTSum, TRIMMER, TR-EduVSum, Multi-Video Positional Bias, HAS, PIGEONS, AdaFrame, and the stub repositories for TRIM, ClipSum, and QEVA. Complete FastPerson/QA-FastPerson and HABSS preprocessing audits, and obtain full texts for publisher-gated SAFCN, CARLSVS, JCDL 2025, and M2UR records. Expand medical/procedural and preference-learning coverage. Resolve TRINITY and SGWIB source conflicts and TripleSumm's appendix number/captioner inconsistencies. Reproduce a controlled supervised baseline and a reward-based baseline using one public split manifest, fixed checkpoint rule and fully specified decoder. Confirm or replace unavailable artifacts without erasing their provenance.

On each meaningful run, update this ledger or add a dated linked audit, update the changelog, preserve old verification dates for untouched claims, and regenerate catalogs. Record exclusions and negative search outcomes as well as additions.
