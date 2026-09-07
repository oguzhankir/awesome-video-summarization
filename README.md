# Awesome Video Summarization

A research handbook and practical learning guide to video summarization: classical and supervised methods, weak/semi/self-supervision, unsupervised and reinforcement learning, training-free selection, multimodal foundation models, and query or user-conditioned systems. It covers keyframes, keyshots, textual and multimodal summaries, with source-linked code, datasets and protocol-aware benchmarks.

**Last literature audit: 2026-09-08.** Coverage is selective and dated; [the ledger](docs/15-coverage.md) distinguishes newly verified entries from inherited technical audits. Source availability does not imply successful reproduction.

**Start learning:** [Learning path](docs/16-learning-path.md) · **Find a paper:** [Catalog](docs/generated/papers.md) · **Find code:** [Implementations](docs/13-implementations.md) · **Compare results:** [Benchmarks](docs/12-benchmarks.md)

## Read the handbook

| Foundation | Method families | Practice and research |
|---|---|---|
| [History and field overview](docs/00-overview.md) | [Classical and supervised](docs/06-classical-supervised.md) | [Task-specific settings](docs/11-task-settings.md) |
| [Taxonomy and formulations](docs/05-taxonomy.md) | [Reconstruction and generative](docs/04-reconstruction-generative.md) | [Protocol-separated benchmarks](docs/12-benchmarks.md) |
| [Mathematical foundations](docs/01-foundations.md) | [RL and heuristic selection](docs/07-reinforcement-heuristics.md) | [Code, features and weights](docs/13-implementations.md) |
| [Evaluation and failure modes](docs/02-evaluation.md) | [Contrastive and self-supervised](docs/08-contrastive-self-supervised.md) | [Open problems](docs/14-open-problems.md) |
| [Datasets and annotations](docs/03-datasets.md) | [Graphs, attention and Transformers](docs/09-graphs-attention.md) | [Coverage and research gaps](docs/15-coverage.md) |
| [Hands-on learning path](docs/16-learning-path.md) | [Multimodal, VLM and Video-LLM](docs/10-foundation-models.md) | [Changelog](CHANGELOG.md) |

## Choose a task, then a protocol

| Axis | Examples |
|---|---|
| Supervision | Supervised, weak/semi/self-supervised, unsupervised, training-free, zero/few-shot |
| Output | Storyboard, keyshot skim, highlight, text, multimodal, timestamp/event summary |
| Setting | Generic, query-focused, personalized, multi-video, egocentric, long-form, online |
| Mechanism | Clustering, sparse/submodular/DPP, recurrent, reconstruction/GAN, RL, graph/attention, contrastive, VLM/LLM, diffusion |

These axes are independent. The four original unsupervised families remain a [sub-taxonomy](docs/01-foundations.md#3-four-useful-unsupervised-system-families). Supervised methods are core reading, including methods whose frozen encoders or auxiliary contrastive losses can obscure their human-label dependence.

## Datasets and results

SumMe and TVSum support classic importance-to-skim studies; VideoXum supports visual and textual summaries; query and egocentric resources have their own annotations and metrics. MoSu adds behavior-derived multimodal importance targets. QA and grounding resources remain explicitly adjacent. See [dataset cards](docs/generated/datasets.md) for licensing, availability, features and unresolved fields.

A score is meaningful only with its split, training data, features, sampling, segmentation, shot values, budget solver, reference aggregation and metric. The [generated benchmark catalog](docs/12-benchmarks.md) isolates unknown/incompatible protocols and distinguishes author-reported values from reproduced ones. It does not assert a field-wide winner.

Recent starting points include **TripleSumm** (ICLR 2026), **SummDiff**, **LLMVS**, **V2Xum-LLM**, **SD-VSum** and **CoE**. The [modern-method audit](docs/10-foundation-models.md) explains their actual training signals and output differences, including TripleSumm's source-pretraining, fine-tuning and direct-transfer distinction.

## Run a first experiment

No model or dataset download is needed for the synthetic decoder example:

```bash
python3 examples/summary_baselines.py --demo
```

Change shot lengths, scores and budget to see how knapsack and user-reference aggregation affect the output. Continue with the [learning path](docs/16-learning-path.md) and [source-audited resources](docs/generated/resources.md), including official implementations, author presentations, tutorials, feature archives and checkpoints.

## Contribute and maintain

[CONTRIBUTING.md](CONTRIBUTING.md) defines evidence, schemas and validation. Edit registries under [data/](data/), then run `python3 scripts/generate_catalog.py`. The [documentation workflow](.github/workflows/validate.yml) checks schemas, generated files, Markdown, mathematics, Mermaid and tests. The [weekly external-link workflow](.github/workflows/link-audit.yml) reports restricted access separately from missing links.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this handbook with the revision you used, and cite original papers/datasets for their methods and results. The repository is [MIT licensed](LICENSE); linked code, datasets, media and model weights retain their own licenses.
