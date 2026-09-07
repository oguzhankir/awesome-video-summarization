# Learning video summarization, from first skim to audited experiment

This path starts on a CPU with a small example, then introduces video handling, feature files, model families and benchmark auditing. Every stage has a concrete output. Upstream projects and tutorials were inspected on **2026-09-08**; their training environments were not run in this audit. Use the [implementation registry](13-implementations.md) for source commits and reproduction caveats.

## 1. Build the evaluation intuition first

Read [Foundations](01-foundations.md) through the system pipeline and [Evaluation](02-evaluation.md) through reference aggregation. Run the repository's standard-library exercise from its root:

```bash
python3 examples/summary_baselines.py --demo
```

This is an educational implementation of summary construction and baselines, not a reimplementation of a named paper. Inspect [the source](../examples/summary_baselines.py), then work through these exercises:

1. Identify the score sequence, shot boundaries, integer duration budget and final binary summary.
2. Compare constant and random scores under identical shot lengths. Then change shot lengths while keeping scores fixed.
3. Compare mean importance as a knapsack item value with duration-weighted importance. Explain why the selected subset changes.
4. Score one prediction against two different human masks; explain how max-user F1 can exceed mean-user F1.
5. Distinguish uniform temporal selection from random scores decoded through knapsack.

**Deliverable:** a short table of selected intervals, occupied budget and F1, with one sentence explaining each difference. Do this before reading a large neural-network benchmark table.

## 2. Learn the programming and video prerequisites

| Resource | Format and owner | Learn this | Exercise |
|---|---|---|---|
| [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | PyTorch official written tutorials | Tensors, datasets, autograd, losses and optimization | Represent a feature sequence as `T × D`; separate model training from evaluation mode |
| [Introduction to PyTorch](https://docs.pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html) | PyTorch official teaching videos with code | The same ideas through a guided video series | Draw the shapes through a small score predictor and explain the gradient path |
| [FFmpeg filter reference](https://ffmpeg.org/ffmpeg-filters.html) | Official tool documentation | Sampling, selection, trimming, timestamp resetting and concatenation | Produce a short skim from a video you may use; keep an original-time manifest |
| [h5py quick start](https://docs.h5py.org/en/stable/quick.html) | Official Python/HDF5 documentation | Groups, datasets, shapes and dtypes | List one benchmark video's keys without loading its entire feature archive into memory |

Use your installed tool's documentation version where available. Recent PyTorch tutorials teach concepts; they do not guarantee compatibility with the older packages used by historical summarization code.

**Deliverable:** one playable skim and a manifest containing input filename, decoded frame count, frame-rate/time-base information, sampled timestamps and output intervals. For variable-rate video, use timestamps rather than assuming `frame_number/fps` is exact.

## 3. Understand a real dataset before fitting a selector

Choose SumMe or TVSum in the [dataset chapter](03-datasets.md). Read the dataset author's annotation description, then the feature archive's provenance. These are separate sources: common GoogLeNet features were produced and repackaged by later researchers.

Inspect the common schema:

```text
video_i/
  features          (T, D)
  picks             (T,)
  change_points     (S, 2)
  n_frame_per_seg   (S,)
  n_frames          scalar N
  user_summary      (U, N)
  gtscore           (T,)     human-derived: isolate from unsupervised training
```

Check that sampled positions are ordered and in range, shot lengths agree with boundary conventions, and human masks use the intended original timeline. Do not rename mismatched arrays into apparent compatibility. The original dppLSTM release uses another schema, and VideoXum/QFVS use different targets and formats.

Read [TVSum's original MATLAB evaluator](https://github.com/yalesong/tvsum/blob/7bf3fb8ca032f5f4b42e22ba41f5e2490b819f96/matlab/script_evaluate_result.m) beside [DR-DSN summary construction](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce/blob/041606fec8e2ba7f4c4900064a76d965a473e53c/vsum_tools.py). Explain how fixed temporal units and inherited KTS boundaries change the reference summaries.

**Deliverable:** a one-video data card with feature owner, backbone, checkpoint, sampling, shape, boundary convention and reference aggregation. Mark unresolved fields as unknown.

## 4. Study one selector family at a time

| Route | Prerequisite resource | Source-reading exercise | Completion criterion |
|---|---|---|---|
| Reconstruction and adversarial learning | [Reconstruction chapter](04-reconstruction-generative.md) | Trace SUM-GAN-AAE selector, reconstructor, discriminator and losses | Explain what each network observes and why soft weighting is not hard subset selection |
| Policy gradients | [Spinning Up: policy optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) | Derive the sampled-action objective, then trace DR-DSN's reward and moving baseline | State how diversity, coverage and length affect the selected subset without human target labels |
| Diversity by determinants | [DPPy documentation](https://dppy.readthedocs.io/en/latest/) | Build a small positive-semidefinite similarity kernel; inspect dppLSTM as a supervised historical example | Distinguish DPP sampling, likelihood, approximate MAP and knapsack |
| Attention | [PGL-SUM source and audit](13-implementations.md#policy-gradient-and-supervised-comparisons) | Trace global/local attention to one score per time step | Name the supervised target and checkpoint-selection procedure |
| Foundation embeddings | [CLIP author code and notebook](https://github.com/openai/CLIP) | Compare normalized image/text similarity for several prompts on fixed frames | Explain query relevance versus generic importance; preserve author preprocessing |

After each route, reuse the same baseline/evaluation pipeline. An architecture improvement cannot be assessed if it also silently changes the feature backbone or budget solver.

**Deliverable:** an architecture sketch plus a list of input tensors, learning signals and output transformations. If running source, record the exact commit, environment, device, seeds, checkpoint and all compatibility changes.

## 5. Reproduce a narrow experiment

Start with one dataset, one known split and one model. A source-reading study is already useful; label an execution attempt separately from a completed reproduction.

1. Choose an implementation with artifacts you can actually retrieve. PGL-SUM, VASNet, DSNet and VideoXum advertise pretrained models; availability and compatibility still require checking.
2. Read the relevant caveats in [the implementation audit](13-implementations.md#3-read-these-source-paths-before-training). For example, VASNet's test-named split influences epoch selection, DSNet's raw-video sampler counts decoded frames, and VideoXum's F1 threshold can exceed a nominal 15% budget.
3. Match the features, split files and summary construction before comparing a paper number.
4. Fix checkpoint selection before evaluating final-test labels. Save full score arrays and masks so metrics can be recomputed.
5. Report all planned seeds, including failures. Include constant/random/uniform baselines and the distribution of occupied summary duration.

**Deliverable:** a reproducibility record another person can execute. If the archived environment cannot be restored, report the precise failure and adapted environment; do not silently relabel the port as the original system.

## 6. Extend to multimodal and query-focused tasks

[VideoXum](https://github.com/jylins/videoxum) teaches paired visual/textual output: inspect its author-released [feature/annotation listing](https://huggingface.co/datasets/jylins/videoxum/tree/main) and [weight listing](https://huggingface.co/jylins/vtsum_blip/tree/main). Keep visual F1, rank statistics, caption quality and cross-modal alignment as separate measurements. Its supervised setting is distinct from using CLIP features without fitting on summary labels.

For long query-focused video, inspect [UniVTG QFVS code](https://github.com/showlab/UniVTG/tree/32659ac7aeba21742a63274f30eba785fc57e247) and the [FCSNA-QFVS notebook](https://github.com/srkds/Query-Focused-Long-Video-Summarization). The latter is an author research notebook with incomplete setup and hard-coded paths. Query-focused semantic matching is not the generic SumMe temporal-overlap metric. A text-only transcript summarizer solves another task again.

**Deliverable:** a task card stating who supplies the query, which labels train the system, whether audio/text are available at inference, the output unit and the exact matching rule.

## 7. Learn to question the score

Use [Rethinking the Evaluation of Video Summaries](https://github.com/mayu-ot/rethinking-evs) for author analysis scripts and notebooks. Revisit one experiment while fixing predicted scores and changing only segmentation. Then compare scores before hard decoding using rank correlation, with its annotation-aggregation rule recorded explicitly.

A useful final project answers one bounded question, for example: “How much does mean-valued versus duration-weighted knapsack change the ranking of these two fixed selectors?” Publish the inputs, seeds, masks and metric code used to answer it. This gives a new learner something concrete to inspect, rerun and challenge.
