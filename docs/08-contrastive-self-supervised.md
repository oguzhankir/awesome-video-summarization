# Contrastive and self-supervised learning

[Home](../README.md) · [Taxonomy](05-taxonomy.md) · [Modern systems](10-foundation-models.md)

Self-supervision describes a learning signal, not proof of label-free deployment. An encoder can be pretrained without summaries and then fine-tuned with them. Record upstream labels, target supervision, prompt/calibration data, and checkpoint selection separately.

## Contrastive selection

[CSUM: Learning to Summarize Videos by Contrasting Clips](https://arxiv.org/abs/2301.05213) uses a differentiable top-k selector to contrast selected features; its unsupervised route and optional personalized labels are separate settings. It is distinct from **CSNet**, which retains adversarial reconstruction.

For a query representation $`z`$, positive $`z^+`$ and candidate keys $`z_j`$, the following **canonical InfoNCE abstraction** illustrates the mechanism:

```math
\mathcal L_{\mathrm{NCE}}=-\log
\frac{\exp(\mathrm{sim}(z,z^+)/\tau)}
{\sum_j\exp(\mathrm{sim}(z,z_j)/\tau)}.
```

The positive appears in the denominator. Positive/negative construction determines what the objective preserves: two views of a whole video encourage instance identity; adjacent clips may encourage temporal consistency; aligned audio/video pairs encourage cross-modal agreement. None automatically implies narrative importance. [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) is a primary source for the contrastive predictive framework, not a summarization benchmark.

## Distillation and representation learning

[SELF-VS](https://arxiv.org/abs/2303.15993) pretrains a Transformer through representation matching to a video-classification CNN. The teacher's classification supervision is part of feature provenance. Its paper must be consulted for downstream training and rank evaluation; a self-supervised encoder does not make a supervised selector unsupervised.

[A2Summ](https://openaccess.thecvf.com/content/CVPR2023/html/He_Align_and_Attend_Multimodal_Summarization_With_Dual_Contrastive_Losses_CVPR_2023_paper.html) combines multimodal alignment and supervised summarization. Contrastive losses are secondary mechanisms when human importance targets also train the selector.

## Masked and predictive objectives

A **repository-level masked-prediction abstraction** is

```math
\mathcal L_{\mathrm{mask}}=\frac1{|M|}\sum_{t\in M}
\|g_\theta(X_{\bar M})_t-x_t\|_2^2.
```

Mask set $`M`$ defines prediction targets, and $`X_{\bar M}`$ denotes the visible input. Prevent an attention path from seeing the target features through another branch. Reconstruction accuracy measures predictable content; rare salient events can remain poorly modeled. [VideoMAE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html) is an encoder resource, not a standalone video summarizer. SUM-SR's [mask-pretraining ablation](04-reconstruction-generative.md#12-sum-sr-discriminator-free-iterative-reconstruction) is distinct from its main five-iteration configuration.

## Reproduction decisions

Publish positive/negative pair construction, temporal overlap exclusions, augmentation settings, memory bank contents, temperature, selected-set cardinality, and any labeled adaptation set. Avoid claiming a mutual-information guarantee for an arbitrary contrastive loss: the bound depends on sampling assumptions. Evaluate selection quality on held-out videos and retain the complete final decoder.
