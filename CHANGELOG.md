# Changelog

Record substantive handbook, evidence, implementation-resource and validation changes on each maintenance run. Dates describe repository updates; individual source-verification dates remain in the registries. An entry does not imply a model was reproduced.

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
