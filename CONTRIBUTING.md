# Contributing

This repository accepts evidence-backed additions and corrections. A paper title plus a repository URL is not a complete entry.

## Paper-entry schema

Copy and fill every field that applies:

```markdown
### Full paper title

- **Citation:** Authors. “Title.” Venue, year. DOI/arXiv/official proceedings.
- **Primary paradigm:** one of the four repository paradigms
- **Supervision:** unsupervised / self-supervised / training-free / zero-shot / weak-external / supervised comparator
- **Input unit and sampling:** frame or clip; exact rate/window/stride
- **Backbone:** architecture, checkpoint, layer, dimension, preprocessing
- **Temporal/fusion module:** selector and context mechanism
- **Losses or rewards:** exact equations or primary-source section references
- **Summary construction:** segmentation, score expansion, shot pooling, solver, budget
- **Evaluation protocol:** dataset, split IDs/regime, reference aggregation, metrics, number of runs
- **Reported result:** value copied from a named primary-source table
- **Code:** official/community/not released; framework; tested commit or release
- **Weights/features:** direct artifact URL, provenance label, checksum if available
- **Known reproduction gaps:** unresolved dependencies, dead links, changed features, or protocol ambiguity
```

## Dataset and feature-entry schema

Record:

1. dataset authors' canonical page and paper;
2. raw-video and annotation URLs separately;
3. number of videos, total and per-video duration, split, task, annotation unit, annotator count, and license/access conditions;
4. each feature artifact's owner, backbone/checkpoint, temporal sampling, tensor dimension, data format, completeness, and checksum;
5. link status and last verification date.

Never label a third-party archive “official.” Do not infer that a feature exists because a paper reports using that backbone.

## Metric-entry schema

Every numeric result must identify:

- frame/shot/keyframe unit;
- temporal matching or overlap rule;
- reference aggregation (`max`, `mean`, or another explicit operation);
- KTS or other segmentation configuration;
- mean versus duration-weighted shot value;
- exact 0/1 knapsack, greedy selection, or thresholding;
- budget and rounding convention;
- split protocol and any augmented/transferred training data;
- mean, deviation/confidence interval, and number of seeds when available.

If a source writes “C-F1,” reproduce its equation and define “coverage.” The acronym alone is insufficient because it is not standardized across this literature.

## Pull-request audit

- [ ] Primary sources support factual claims.
- [ ] Reported numbers include their original protocol.
- [ ] Code is labeled official or community correctly.
- [ ] Artifact links point to files/pages, not search results.
- [ ] Missing code, weights, or feature tensors are stated as missing.
- [ ] Links were checked and the date recorded.
- [ ] No human-derived field leaks into an “unsupervised” training or model-selection path.
- [ ] New tables remain readable on GitHub without horizontal HTML layouts.
