# Changelog

Record substantive handbook, evidence, implementation-resource and validation changes on each maintenance run. Dates describe repository updates; individual source-verification dates remain in the registries. An entry does not imply a model was reproduced.

## 2026-10-25

### Added in 2026-10-25

- **RVLSM**, first posted 2026-10-19, plus **Semantic-Guided Unsupervised Video Summarization** (January 2026) and **SoccerHigh** (MMSports 2025) as explicitly dated backfill: 76 papers in total.
- SoccerHigh dataset/baseline cards and a pinned LVSum evaluator resource: 31 datasets and 46 resources. No numerical result rows added; the 88 existing rows retain their original numerical evidence boundaries.
- A [dated source audit](docs/audits/2026-10-25-weekly.md) with actual queries, candidate exclusions, upstream runtime blockers and network limitations.

### Fixed in 2026-10-25

- Corrected inherited paper identities, arXiv links, supervision, architectures and dataset counts throughout the handbook. TRIM/TRIMMER are trained visual methods; VASTSum is supervised uncertainty/decoder-aligned learning; ClipSum generates YouCook2 text; TR-EduVSum uses Turkish human summaries; LVSum has 72 videos.
- Replaced false TRIM/ClipSum stub claims with source inventories and concrete execution caveats. Updated KnowVis to its released pipeline, LVSum to public annotations/evaluators and separate licenses, QEVA to its canonical README-only repository, and FramePack-U to its Apache-2.0 release.
- Corrected SD-MVSum's author-linked release to Zenodo 19919955 and its loader key descriptions. Four inherited SD-MVSum result rows now explicitly isolate unverified archive-to-paper split identity; their score values are unchanged.
- Withdrew unsupported SummDiff evaluation-script correction claims and the unrelated positional-bias formulas. Earlier changelog prose below is historical and is superseded by these source corrections.
- Regenerated all catalog views and repaired the latest-audit links. Preserved the existing weekly-branch URL-alignment commit while integrating the deeper source corrections.

### Audit gaps in 2026-10-25

- No upstream training, checkpoint execution, paid API call or large media/feature archive download. ClipSum has missing unconditional imports; TRIM has CUDA/path/dependency and evaluation caveats. QEVA remains a stub.
- The external-link attempt reported retry/unresolved for all 495 discovered URLs; the runner's external DNS failures prevent treating this as link verification. A second final-tree pass is recorded in the dated audit.
- Search was targeted, not an exhaustive venue census. CVPR 2026 AgenticVS full text was access-restricted; Minimal Clips remains a follow-up candidate. Existing numerical results were not comprehensively re-audited.

## 2026-10-18

### Added in 2026-10-18

- Sixteen verified paper records from the 2026-10-01–2026-10-18 literature scan and 2025–2026 core backfill: **SD-MVSum**, **TRIM**, **TRIMMER**, **VASTSum**, **LVSum**, **TR-EduVSum**, **ClipSum**, **Multi-Video Positional Bias**, **HAS**, **PIGEONS**, **FramePack-U**, **AdaFrame**, **QEVA**, **SAFCN**, **CARLSVS**, and **JCDL 2025 historical video archive segmentation/summarization** (bringing the registry from 57 to **73 papers**).
- Five dataset/benchmark cards: **SM-VideoXum**, **SM-Mr. HiSum**, **LVSum**, **MLVU(VS)-Eval**, and **TR-EduVSum** (bringing the registry from 25 to **30 datasets**).
- Ten pinned implementation and stub resource cards: **SD-MVSum** (`e88adbf3`), **CSTA** (`b070f7d9`), **A2Summ** (`5c7dceac`), **V2Xum-LLM** (`6b7c06d0`), **SummDiff** (`1d4b8ae7`), **SUM-SR** (`fe22076f`), **FramePack-U** (`aed28d7d`), **TRIM stub** (`86a6be3e`), **ClipSum stub** (`c3746c1b`), and **QEVA stub** (`657cad6b`) (bringing the registry from 34 to **44 resources**).
- Twenty-eight protocol-isolated primary-source benchmark rows across **CSTA**, **VASTSum**, **TRIM**, **TRIMMER**, **SD-MVSum**, **QEVA**, and **LVSum** (bringing the registry from 60 to **88 results**).

### Changed in 2026-10-18

- Completed primary-source PDF and repository verification for **CSTA**, **CSUM**, **SELF-VS**, **QEVA**, **A2Summ**, **V2Xum-LLM**, **Prompts to Summaries**, **Context-Aware Pseudo-Label Scoring**, **SummDiff**, and **SUM-SR**, replacing `"Not independently verified"` protocol placeholders with exact primary-source specifications and pinning repository commit SHAs.
- Updated `README.md`, `docs/02-evaluation.md`, `docs/03-datasets.md`, `docs/08-contrastive-self-supervised.md`, `docs/09-graphs-attention.md`, `docs/10-foundation-models.md`, `docs/11-task-settings.md`, `docs/13-implementations.md`, and `docs/15-coverage.md`, and regenerated all four catalogs (`docs/generated/*.md` and `docs/12-benchmarks.md`).

### Fixed in 2026-10-18

- Recorded CSTA's two post-publication rank-correlation fixes on SumMe (`0.205/0.233` vs. paper Table 2 `0.246/0.274`) and TVSum (`0.194/0.255`), and SummDiff's repository note on updated rank-correlation scripts (`0.33/0.46` TVSum, `0.29/0.35` SumMe vs. paper Table 1 `0.21/0.27` and `0.22/0.25`).
- Distinguished SD-MVSum's resolved HDF5 schema (`gtsummaries` on SM-VideoXum; `gtscore` and `gtsummary` on SM-Mr. HiSum) from SD-VSum's earlier `gtscores` loader mismatch, and isolated TRIM's 5-fold 20% test split averages from its full-dataset (all 25/50 videos) evaluation tables.

### Audit gaps in 2026-10-18 review

- No upstream GPU training, paid API call, or large dataset/checkpoint binary download was executed. Full texts for publisher-gated IEEE/Elsevier records (**SAFCN**, **CARLSVS**, **JCDL 2025**) remain restricted, and public code/dataset uploads remain pending for **LVSum**, **VASTSum**, **TRIMMER**, **TR-EduVSum**, **Multi-Video Positional Bias**, **HAS**, **PIGEONS**, **AdaFrame**, and the stub repositories for **TRIM**, **ClipSum**, and **QEVA**.

## 2026-10-01

### Added in 2026-10-01

- TSMD, EviDETR and MultiVENT-Raw paper records; 16 protocol-isolated TSMD robustness rows and two QVHighlights highlight rows. No generic SumMe/TVSum score was added.
- Dataset cards for Mr. HiSum, QVHighlights, MultiVENT-Raw and MicroVENT, plus the pinned TSMD implementation record.
- A dated arXiv/source audit for the 2026-09-20–10-01 delta, with candidate exclusions, gated-access notes and verified publication boundaries.

### Changed in 2026-10-01

- Updated the dataset, foundation-model, task-setting and coverage chapters and README to describe query-conditioned highlights, modality-missingness robustness and multi-video cited reports.
- Regenerated paper, dataset, result and implementation catalogs from the registries.

### Audit gaps in 2026-10-01 review

- No model, GPU run or large video/feature archive was executed or downloaded. The arXiv identity-centric English version remains deferred because its claimed earlier Spanish journal record and conflicting metric statements were not resolved. No external-link audit was repeated during the literature scan; the prior 370-URL run remained unresolved due to environment DNS failures.

## 2026-10-02

### Fixed in 2026-10-02

- Replaced a stale Kingston research-profile URL for VASNet's cited paper with its publisher DOI after the link audit returned 404.
- Made the 2026-10-01 changelog headings date-specific after the validation workflow flagged duplicate Markdown anchors.

## 2026-09-19

### Added in 2026-09-19

- Ten recent records from the 2026-08-09 overlap window: TRINITY, KnowVis, SGWIB, metadata-conditioned zero-shot highlights, Audio for Sports Highlight Detection, Semantic Action Graph, Unified Agentic Video Editing, the Wiley vision-language/hierarchical model, the Springer soccer pipeline and metadata-only M2UR.
- Five historical gap records: SSPVS, CLIP-It!, Multi-VidSum, personalized day-long egocentric summarization and causal online diversity sampling.
- TRINITY, KnowVis and Multi-VidSum dataset cards; four protocol-isolated Byra et al. Table 1 correlation records; twelve pinned implementation/evaluator resources. The registries now contain 54 papers, 21 datasets, 38 benchmark rows and 33 resources.
- A dated search/source audit covering the 30-day overlap, 11,568-record official arXiv scan, publisher/proceedings follow-up, exclusions, restricted sources, code inspection and remaining gaps.

### Changed in 2026-09-19

- Updated the foundation-model, task-setting, dataset, self-supervision, RL, implementation and learning-path chapters with recent methods, historical coverage and runnable audit exercises.
- Extended the paper schema with an explicit `not reported` supervision state so restricted metadata-only records do not require invented labels.
- Pinned and inspected TripleSumm, LLMVS, CoE, SD-VSum, TRINITY, KnowVis, SSPVS, CLIP-It, Multi-VidSum and personalized-egocentric source releases without downloading large artifacts.
- Replaced the generic live CLIP source reference with commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.

### Fixed in 2026-09-19

- Separated Byra et al.'s TVSum text+image+style variant from its SumMe text-only/category-bootstrap variant; rejected the paper's shared “best variant” label as one configuration and withheld underspecified top-5 mAP.
- Withheld SGWIB values because MoSu prose conflicts with Table III; isolated TRINITY's paper/README/manifest counts and optimizer/epoch differences; recorded KnowVis as a release stub rather than working code.
- Recorded concrete release blockers: LLMVS test-influenced checkpoint selection, unused visual features and overlapping partitions; CoE dependency/configuration mismatches and disabled refinement; SD-VSum HDF5 key and loader-routing errors; TRINITY baseline signature/dimension mismatches; SSPVS test-influenced epoch selection.
- Corrected result bounds so the actual `Temporal-overlap F1` / `Frame-overlap F1` plus `percent` records are checked in `[0, 100]`, and correlation coefficients are checked in `[-1, 1]`.
- Removed reported variance from the benchmark protocol-group key: uncertainty remains displayed with each result but can no longer split otherwise identical protocols. Unknown protocol metadata still isolates rows.

### Remaining audit work after 2026-09-19

- No upstream GPU model, paid API, large feature/video archive or published score was executed. M2UR's IEEE PDF remained restricted; HABSS, FastPerson/QA-FastPerson, QEVA, multilingual, multi-camera, medical/procedural and preference-learning audits remain follow-ups. No global-best or exhaustive-coverage claim is made.
- The full 370-URL link audit was inconclusive because every endpoint hit the same execution-environment DNS failure; all remained unresolved and no citation was removed.

## 2026-09-08

### Added

- Multi-axis taxonomy, historical overview, classical/supervised, RL, contrastive, graph/attention, multimodal/foundation-model, specialized-setting, open-problem and learning-path chapters.
- 39 unique paper records: 27 source-reviewed in this revision, plus 12 migrated reconstruction records retaining their earlier technical-audit date. New source reviews include TripleSumm, its MoSu/transfer setting, LLMVS, V2Xum-LLM, CoE, SD-VSum and core supervised/history references.
- 34 primary-source benchmark records with exact table/page provenance, training regime, metrics and explicit unresolved split identities; 18 dataset/annotation/adjacent-resource records; 21 code/tutorial/tool/artifact records.
- Source inspection of 12 implementation/evaluation repositories at pinned commits; feature/checkpoint and learning-resource links, license and environment caveats. Upstream models and archives were not executed/downloaded.
- JSON Schemas, generated catalogs, Markdown/link/math/Mermaid validation, GitHub HTML inspection utility, GitHub Actions workflows and exact known-link-issue tracking.
- A standard-library Python decoder/F1 demonstration and regression tests, including exhaustive knapsack verification.
- Citation metadata and dated coverage, source and overhaul audit ledgers.

### Changed

- Renamed the landing page to **Awesome Video Summarization** and made supervised methods and textual/multimodal outputs first-class subjects.
- Reduced README to navigation and entry points; retained mathematical depth in focused chapters. The former four-category taxonomy is now an unsupervised sub-taxonomy.
- Extended dataset coverage with query-specific, multimodal behavioral and long-video text/annotation resources; distinguished core summarization from QA, grounding and highlights.
- Kept inherited archive measurements and checksums explicitly dated rather than presenting them as newly downloaded or revalidated artifacts.

### Fixed

- Replaced 23 operator macros in the original chapters; repaired an unsupported inline math macro and a raw table-pipe delimiter. GitHub HTML exposed three formulas misparsed as headings and lost norm/brace/spacing escapes. Converted 105 display blocks and 233 inline expressions to GitHub's protected math syntax and added regression checks.
- Restored the missing external-link workflow and aligned README claims with the actual workflows.
- Corrected a VASNet source-file path and documented two confirmed missing historical Box archives.
- Recorded VideoXum's inclusive top-15% threshold and mean-reference correlation code; recorded VASNet and DSNet test-key checkpoint selection and Zhang's TVSum reference-budget discrepancy.
- Resolved “Triple-Sum” to **TripleSumm** and separated source pretraining, target fine-tuning and direct transfer. Preserved conflicting appendix prose and captioner naming as unresolved source issues.
- Prevented unknown split identities, malformed links, invalid math syntax and nonfinite result values from silently passing validation.

### Remaining audit work

- No field-wide SOTA claim, GPU training reproduction, full artifact-integrity check or exhaustive venue census is made. Browser Preview, restricted publisher pages, legacy host failures and narrower domain/streaming coverage remain documented follow-ups.

## 2026-09-06 — inherited content audit

The previous handbook contained detailed foundations, evaluation, datasets and reconstruction/generative material. Its exact archive statistics, benchmark transcriptions and source-availability findings are retained with their original audit scope unless explicitly rechecked above. This entry describes inherited document evidence, not a newly reconstructed release history.
