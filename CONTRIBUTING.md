# Contributing

Add evidence-backed improvements to the handbook, implementation resources, or reproducibility tooling. Supervised, weakly supervised, semi-supervised, unsupervised, self-supervised, training-free, zero-shot and few-shot work are all in scope.

## Choose the right record

| Contribution | Edit | Generated view |
|---|---|---|
| A paper and its documented variants | A JSON array in `data/papers/` | [Paper catalog](docs/generated/papers.md) |
| A protocol-specific result | `data/results.json` | [Benchmarks](docs/12-benchmarks.md) |
| A dataset or annotation layer | `data/datasets.json` | [Dataset catalog](docs/generated/datasets.md) |
| Code, tutorials, tools, features or weights | `data/resources.json` | [Resource catalog](docs/generated/resources.md) |
| Architectural explanation or a correction | Relevant chapter in `docs/` | Handwritten explanation |

Schemas in [schemas/](schemas/) define every required field and allowed categories. Use `Not reported` when the inspected primary source omits a detail, `Not released` only when justified by the author release, and `Not independently verified` when the audit did not establish it. Do not turn an unchecked field into a negative claim. A verification date applies to the declared level of evidence; it never implies training or reproduction.

## Evidence and classification

Read the primary paper and author-linked code. Store exact titles, full authors, archival venue/year, preprint version when material, URLs and field-specific limitations. A paper with several variants remains one paper record, while results identify the variant through `method`. Separate supervision, output, task and mechanism tags. A frozen backbone, contrastive auxiliary loss or “zero-shot” title does not override the actual label usage.

For every number, identify the exact source table and page, dataset, training regime, split, feature/sampling configuration, segmentation, score pooling, budget, solver, user-reference aggregation, metric, runs and uncertainty. `split_identity` must identify a verified shared artifact or explicit experiment; unknown identity isolates the row. Do not rank independently sampled splits as equivalent just because both say “80/20.” Values are numeric, units explicit, and nonfinite numbers forbidden. Do not copy aggregator leaderboards when the primary table is available.

Official code needs author/paper attribution. Record the inspected commit, dependencies, preprocessing, feature schema, weights, evaluation implementation, license and execution status. Existence is not proof of a working reproduction. Do not bulk-vendor third-party code or weights into this repository. For a local educational example, label its assumptions and test the behavior that matters.

Dataset records distinguish source videos, annotation layers, derived features and adjacent tasks. Keep raw media/annotation links separate; label mirrors. Record backbone/checkpoint, dimension, sample rate, dtype, normalization and checksums when verified. Human-derived fields colocated in HDF5 must not silently enter unsupervised optimization or model selection.

## Validate a change

From the repository root, use Python 3.12 and Node.js 22:

```bash
python3 -m pip install -r requirements-dev.txt
npm ci --ignore-scripts --no-audit --no-fund
python3 scripts/generate_catalog.py
python3 scripts/validate.py
npm run check:render
python3 -m unittest discover -s tests -v
python3 scripts/audit_links.py --output /tmp/avs-link-report.json
git diff --check
```

The link audit only reads response headers. HTTP 401/403/429 indicate restricted access; 404/410 indicate missing targets; DNS, timeout and server errors require investigation. Exact documented historical failures in [.github/known-link-issues.json](.github/known-link-issues.json) are still requested on each run and reported as known missing; unexplained new failures fail the check. Keep access restrictions visible instead of deleting a citation. Run tools in a disposable environment for untrusted contributions; inspect code changes before executing them.

The renderer checks equations with KaTeX and Mermaid with its actual parser. GitHub has a separate rendering pipeline. Inspect its Preview for math, tables, anchors and diagrams before merging. Avoid unsupported macros; use `\mathrm{...}`. Prefer fenced `math` blocks and backtick-protected inline math, as documented in [GitHub's mathematical-expression guide](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions), to preserve LaTeX escapes through Markdown processing. If using standalone `$$` lines, keep blank lines around the block and never put a bare equals or minus sign on its own formula line. See the [rendering audit](docs/audits/2026-09-08-overhaul.md) for actual checks and limits.

## Record the work

Update [CHANGELOG.md](CHANGELOG.md) on each meaningful maintenance run with Added, Changed, Fixed and audit gaps as appropriate. Extend [the coverage ledger](docs/15-coverage.md) with the queries, primary candidates, exclusions and access failures actually observed. Regenerate catalog views; do not hand-edit generated output. Changes that find nothing new do not need an empty commit or a falsely advanced literature-audit date.

Use a feature branch and concise English commits with DCO sign-off (`git commit -s`). Do not conflate DCO with a cryptographic/GPG signature. The maintainer reviews and pushes local maintenance branches and opens PRs.
