# Open problems and research directions

[Home](../README.md) · [Coverage and unresolved candidates](15-coverage.md)

These are research questions and suggested experiments, not verified conclusions about every method.

## Evaluation that isolates selection quality

[Otani et al.](https://arxiv.org/abs/1903.11328) demonstrate how segmentation and shot decoding affect F1. A useful next experiment holds boundaries, budget solver and features fixed, reports raw score correlations alongside the assembled skim, and evaluates whether a model improves over constant/random scores. Run the same study under mean and max reference aggregation rather than treating either as universal.

## Preference distributions and controllability

Averaged labels hide disagreement. [SummDiff](https://arxiv.org/abs/2510.08458) models individual score distributions; [SD-VSum](https://arxiv.org/abs/2505.03319) conditions on a desired script. Test whether diverse outputs preserve validity and factual content, and whether the same user receives consistent control across videos. Evaluate calibration separately from diversity.

## Language grounding and omitted evidence

[V2Xum-LLM](https://arxiv.org/abs/2404.12353) and [CoE](https://arxiv.org/abs/2603.06213) make textual or multimodal summaries practical research targets. Test timestamp hallucination, entity attribution, event order, and important visual events absent from ASR. A fluent summary can still omit the decisive scene. Compare source-grounded human review with automatic text similarity and disclose judge model/version when an LLM is used for evaluation.

## Upstream data and model-selection leakage

[TripleSumm](https://arxiv.org/abs/2603.01169) illustrates separate source pretraining, target fine-tuning, and long-video transfer. Publish target overlap checks, split identities, prompt examples and all validation choices. For pretrained encoders whose upstream data is undisclosed, record contamination as unknown. Test-set-based checkpoint choice deserves a separate result label even when training gradients never read the test labels.

## Long-video cost and streaming constraints

Sampling can erase brief events; hierarchical memory can lose long-range causal links. Measure decoder, encoder, captioning, fusion and selection time separately. Compare peak memory, wall time, API cost and summary quality at matched budgets. A causal online experiment must restrict future access and name its revision/look-ahead policy; full-video preprocessing is an offline advantage.

## Datasets beyond the classic pair

The [dataset registry](03-datasets.md) separates skims, keyframes, behavior-derived highlights, text summaries, query-specific annotations, and adjacent QA resources. Test robustness across domains and annotator populations before claiming broad summarization ability. Medical/procedural, multilingual, multi-camera and accessibility-focused evaluations need deeper primary-source coverage in later revisions.

## Reproduction as a research contribution

Rebuilding one legacy baseline with frozen data hashes, evaluator tests, checkpoint selection, seeds, and preprocessing is useful even without a new network. Record original and corrected outputs when fixing a protocol bug. The [source audit](13-implementations.md) provides concrete starting points, including feature schemas and known environment gaps.
