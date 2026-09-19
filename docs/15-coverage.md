# Coverage methodology and research gaps

[Home](../README.md) · [Paper catalog](generated/papers.md) · [Implementation sources](generated/resources.md) · [Latest audit](audits/2026-09-19-weekly.md) · [Overhaul audit](audits/2026-09-08-overhaul.md)

**Literature/source review: 2026-09-19.** This is a selective, evidence-backed handbook, not a systematic review with a completed census of every proceedings volume. A registry can have complete fields while experiment details remain explicitly unverified. Neither a verification date nor a reachable repository establishes successful ML reproduction.

## 1. Inclusion and verification

A core method must solve a defined summarization target; an adjacent resource must explain what it contributes and what target it lacks. Verify title, authors, venue/year and release against the primary publication. Follow paper/project links to author code, then inspect implementation and artifacts separately. A method with multiple variants counts as one publication and multiple result records.

Every new number comes from an identified primary table/page. Exact split identities, boundary files, reference construction and checkpoint selection are recorded or explicitly left unknown. In this release unknown split identities isolate results; no compatible global best result is established. There are no independently reproduced model scores.

Inherited reconstruction cards retain their earlier 2026-09-06 technical-audit dates unless a particular entry was rechecked. The 2026-09-08 overhaul records also retain their field-level dates unless this weekly audit changed them. Exact legacy archive durations and checksums are labeled inherited and were not recomputed from downloaded bytes. No model score was independently reproduced.

## 2. Executed search matrix

The latest pass used a 30-day overlap from 2026-08-09 and searched through 2026-09-19. The official arXiv submitted-date scan covered 11,568 records across `cs.CV`, `cs.MM`, `cs.CL`, `cs.AI` and `cs.HC`, supplemented by updated-record queries, Crossref, publisher pages, proceedings lists, paper references and GitHub. This table states actual depth rather than suggesting every venue was exhaustively searched.

| Venue/source family | Work performed | Limits |
|---|---|---|
| CVPR, ICCV, ECCV / CVF and ECVA | Landmark supervised/reconstruction methods; CSTA, A2Summ, LfVS, LLMVS, SummDiff, CoE, TRINITY and task-specific sources | Selected papers and citation trails; TRINITY's author acceptance claim lacks an independently indexed proceedings entry |
| NeurIPS | seqDPP and DPP/representation-learning foundations; targeted summarization queries | No complete year-by-year summarization census |
| AAAI | Cycle-SUM, CSNet, DR-DSN and V2Xum-LLM primary sources | Selected primary records |
| ICLR | Exact TripleSumm identity, main/appendix protocols, official artifacts and pinned implementation | Focused 2026 check |
| ACM Multimedia and associated workshops | ACGAN, stepwise adversarial methods, SD-VSum; author code and project evidence | Publisher restrictions on some pages; accepted manuscripts used |
| TIP, TCSVT, TMM, Pattern Recognition | DSNet, AC-SUM-GAN, VideoXum, CAAN and sparse-reconstruction discovery | Paper-level coverage; no journal-wide census |
| TPAMI, WACV, BMVC, ICIP | SSPVS, personalized day-long egocentric summaries, causal streaming and official M2UR metadata | M2UR full text was restricted; no complete venue census |
| ACL, EACL, EMNLP | SummScreen, Multi-VidSum, KnowVis and video-to-text/reference-free candidates | KnowVis proceedings entry is not independently indexed; textual metrics remain separate from keyshot overlap |
| arXiv and author projects | Full five-category date-window scan plus training-free, zero/few-shot, long-form, query, online, personalized, sports and multimodal terms | Preprints remain labeled; indexing delay, later revisions and non-arXiv papers require publisher follow-up |
| Crossref, Wiley, Springer, IEEE and ACM | In-window DOI/publisher discovery for multimodal generic and sports summarizers | Publisher security gates and first-online/issue-date differences are recorded explicitly |
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
video summarization submitted 2026-08-09 TO 2026-09-19 cs.CV cs.MM cs.CL cs.AI cs.HC
video highlight detection audio sports 2026; video summary visual narrative agentic 2026
video summarization personalized multi-video causal streaming online 2026
video summarization site:doi.org 2026 CLIP BLIP-2 hierarchical temporal soccer
```

The deliberately broad DSNet query initially suggested AAAI; the verified archival venue is TIP. Search terms are hypotheses, not evidence. GitHub lookups included exact title/author/identifier searches for every admitted recent paper plus historical source leads. [The 2026-09-08 source ledger](audits/2026-09-08-sources.md) preserves overhaul provenance; [the 2026-09-19 audit](audits/2026-09-19-weekly.md) records this run's search and dispositions.

## 3. Candidates and disposition

| Candidate | Decision and reason |
|---|---|
| Triple-Sum / Triple Sum / Triplet-Sum | Resolved to [TripleSumm](https://arxiv.org/abs/2603.01169); no separate paper established for spelling variants |
| SummDiff | Included as supervised/behavior-supervised diffusion; no longer excluded by the former unsupervised-only scope |
| Context-Aware Pseudo-Label Scoring | Included with label-assisted rubric-calibration caveat; headline comparison numbers withheld because protocol/citation issues remain |
| CoE / Cut to the Chase | Included as text summarization; training-free with five training-set style examples |
| EdgeVidSum | Included as demo/report and personalized playback; not established as a causal streaming benchmark |
| [SSPVS, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Li_Progressive_Video_Summarization_via_Multimodal_Self-Supervised_Learning_WACV_2023_paper.html) | Included after paper/source audit; self-supervision is pretraining and released epoch selection is test-influenced |
| [Multi-VidSum, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.457/) | Included with dataset, official code and separate evaluator; keyframe-caption alignment metrics stay isolated |
| [CLIP-It!, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/7503cfacd12053d309b6bed5c89de212-Abstract.html) | Included; official repository is a stub and the community source is only a forward-pass sketch |
| [QEVA, EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.1340/) | Primary evaluation contribution found; judge/metric and artifact audit deferred |
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

Selected publisher/CVF/author pages rejected automated requests or lacked usable HTML; primary PDFs, arXiv versions or author source were used where available. M2UR's IEEE PDF returned a security challenge and remains a metadata-only record. The old QFVS project could not be retrieved. Legacy VASNet Box artifacts return 404 and remain documented as [known link issues](../.github/known-link-issues.json); they are rechecked, not silently ignored. The required 2026-09-19 link-audit run saw environment-level DNS failure for all 370 URLs and therefore classified every endpoint as unresolved; it could not update the differentiated 2026-09-08 snapshot. Access restrictions, unresolved hosts and missing artifacts are different statuses.

No GPU environments were rebuilt. No datasets, checkpoints or feature archives were downloaded or rehashed. Modern repositories may need large models, a model-serving stack or API access. Dependency and evaluator source were inspected, which supports concrete caveats but not a claim of complete reproduction.

## 5. Next audit priorities

Complete QEVA/reference-free evaluation, FastPerson/QA-FastPerson and HABSS preprocessing audits. Expand multilingual, multi-camera/cross-video, medical/procedural and preference-learning coverage. Resolve TRINITY and SGWIB source conflicts, M2UR's restricted protocol, and TripleSumm's appendix number/captioner inconsistencies. Reproduce a controlled supervised baseline and a reward-based baseline using one public split manifest, fixed checkpoint rule and fully specified decoder. Confirm or replace unavailable artifacts without erasing their provenance.

On each meaningful run, update this ledger or add a dated linked audit, update the changelog, preserve old verification dates for untouched claims, and regenerate catalogs. Record exclusions and negative search outcomes as well as additions.
