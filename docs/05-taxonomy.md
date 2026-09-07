# Taxonomy and problem formulations

[Home](../README.md) · [Mathematical foundations](01-foundations.md) · [Paper catalog](generated/papers.md)

A method is a tuple of supervision, output, setting, and mechanism. A Transformer can be supervised, self-supervised, or frozen; “uses CLIP” specifies none of those by itself. The registries store multiple tags, with the dominant mechanism first and the actual training signal in the objective field.

## Supervision axis

| Regime | Operational test | Example or boundary |
|---|---|---|
| Fully supervised | Do human scores, selected shots, or summaries enter the task loss? | [vsLSTM/dppLSTM](https://arxiv.org/abs/1605.08110), VASNet, SummDiff |
| Weakly supervised | Do indirect labels, web priors, or unpaired edited summaries guide selection? | [UnpairedVSN](https://arxiv.org/abs/1805.12174) |
| Semi-supervised | Are a labeled subset and unlabeled examples used together? | Distinguish the paired-subset UnpairedVSN experiments from its unpaired setting |
| Unsupervised | Does selector training avoid human summary/importance targets? | SUM-GAN, the unsupervised DR-DSN variant |
| Self-supervised | Is the learning target constructed from input views or predictive tasks? | [CSUM](https://arxiv.org/abs/2301.05213); downstream fine-tuning must be tagged separately |
| Training-free | Are task parameters left unfitted during the declared deployment stage? | Frozen prompting can still consume labeled in-context examples |
| Zero-shot | Are no target-task labeled examples used for adaptation or conditioning? | State the source/target boundary and any supervised source pretraining |
| Few-shot | Are a small number of target examples supplied? | Specify their count, selection, and whether they update weights or only the prompt |

“Annotation-free” is stricter than “no gradient updates.” Model selection, prompt choice, normalization, and reference demonstrations can use human labels. Track those channels explicitly. A record with multiple regime tags describes documented variants/stages, not an assertion that contradictory definitions hold for one training run.

## Output axis

| Output | Predicted object | Constraint and evaluation |
|---|---|---|
| Keyframes / storyboard | Ordered frame subset | Frame count; content matching or source-specific semantic metric |
| Keyshots / video skim | Ordered source intervals | Duration budget; temporal-overlap F1 and viewing quality |
| Highlight reel | Salient task/domain moments | Highlight relevance or mAP; a different target from whole-story coverage |
| Textual summary | Generated text | Grounding, factuality, coverage; source-specific text metrics |
| Multimodal summary | Coordinated text and visual selections | Evaluate each branch and agreement between them |
| Structured timestamp / event summary | Events, intervals, descriptions | Schema validity, temporal grounding, event coverage, duration |

## Setting axis

Generic single-video; query-focused; user-conditioned/personalized; multi-video/co-summarization; egocentric/lifelog; long-form; online/streaming; and domain-specific instructional, surveillance, sports, meeting, or medical video are independent settings. A long video is not automatically streaming: a bidirectional model can inspect its future. Domain-specific coverage in this revision is uneven; see the [coverage ledger](15-coverage.md).

## Mechanism axis

Classical clustering, sparse coding, submodular optimization and DPP; recurrent/sequence models; reconstruction/generative learning; adversarial/cycle learning; RL/actor–critic; graph reasoning; attention/Transformers; contrastive/self-supervised learning; audio–visual–text fusion; CLIP/VLM scoring; Video-LLM reasoning; and diffusion are mechanisms. The original four unsupervised families remain useful in [foundations](01-foundations.md#3-four-useful-unsupervised-system-families), but do not cover the entire field.

## Unified formulation

The following is a **repository abstraction**, not one paper's equation. Given video $`V`$, optional query $`q`$, user context $`u`$, and budget $`B`$, predict

```math
S^*=\arg\max_{S\in\mathcal F(V,B)} U_\theta(S;V,q,u).
```

For an extractive skim, $`\mathcal F`$ contains legal source intervals whose total duration is bounded. For text it contains token sequences satisfying a declared length and grounding policy. Learned models minimize a training objective over labeled or constructed targets; frozen systems can estimate utility without fitting task parameters.

A query-conditioned repository abstraction is

```math
U(S;V,q)=\lambda_q\sum_{i\in S}\mathrm{sim}(x_i,q)
+\lambda_c\sum_{t}\max_{i\in S}\mathrm{sim}(x_t,x_i)
-\lambda_r\sum_{i<j,\ i,j\in S}\mathrm{sim}(x_i,x_j).
```

The terms capture relevance, coverage, and redundancy; this is not the exact loss of QFVS or a Video-LLM. Query-aware training and semantic evaluation are motivated by [Sharghi et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Sharghi_Query-Focused_Video_Summarization_CVPR_2017_paper.html). Weights tuned on test annotations change the declared supervision protocol.
