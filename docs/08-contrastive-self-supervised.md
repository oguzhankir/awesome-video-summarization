# Contrastive and self-supervised learning

[Home](../README.md) · [Taxonomy](05-taxonomy.md) · [Modern systems](10-foundation-models.md)

Self-supervision describes a learning signal, not proof of label-free deployment. An encoder can be pretrained without summaries and then fine-tuned with them. Record upstream labels, target supervision, prompt/calibration data, and checkpoint selection separately.

## Contrastive selection

[CSUM: Learning to Summarize Videos by Contrasting Clips](https://arxiv.org/abs/2301.05213) uses a two-layer bidirectional Transformer and a perturbed differentiable Top-$`K`$ operator ($`K=\max(\lfloor rT\rfloor,1)`$) to select representative subsets of 1,024-D GoogLeNet features (`pool5`) and maximize cross-video InfoNCE mutual information ($`\tau=0.07`$), regularized by uniqueness, continuity, diversity, and budget-rate losses. In addition to unsupervised canonical, augmented, and transfer benchmarks on SumMe and TVSum, it evaluates a weakly supervised personalized two-branch extension on the 4-video UTE EgoSum dataset using leave-one-out training and bipartite set-matching F1. It is distinct from **CSNet**, which retains adversarial reconstruction.

For a query representation $`z`$, positive $`z^+`$ and candidate keys $`z_j`$, the following **canonical InfoNCE abstraction** illustrates the mechanism:

```math
\mathcal L_{\mathrm{NCE}}=-\log
\frac{\exp(\mathrm{sim}(z,z^+)/\tau)}
{\sum_j\exp(\mathrm{sim}(z,z_j)/\tau)}.
```

The positive appears in the denominator. Positive/negative construction determines what the objective preserves: two views of a whole video encourage instance identity; adjacent clips may encourage temporal consistency; aligned audio/video pairs encourage cross-modal agreement. None automatically implies narrative importance. [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) is a primary source for the contrastive predictive framework, not a summarization benchmark.

## Distillation and representation learning

[SELF-VS](https://arxiv.org/abs/2303.15993) pretrains a video sequence encoder (two-layer ResNet-18 temporal-convolution trunk plus eight-head Transformer encoder) by matching mean-pooled sequence embeddings to a frozen 3D ResNet-50 teacher pretrained on Kinetics-400 ($`\mathcal{L}_{\mathrm{cos}}=1-\cos(\phi_v,\theta_v)`$). Downstream evaluation adds a single linear layer trained with binary cross-entropy on 5-fold SumMe/TVSum partitions (both canonical and OVP+YouTube-augmented) and evaluates rank correlations ($`\tau=0.184, \rho=0.219`$ on TVSum; $`\tau=0.141, \rho=0.158`$ on SumMe). The teacher's classification supervision is part of feature provenance, and the linear probe remains a supervised downstream stage.

[VASTSum](https://arxiv.org/abs/2605.09507v1) aligns vision, audio, subtitles, and transcripts through a unified four-stream Transformer architecture and trains with a dual-path loss combining regression MSE and listwise Ranking Cross-Entropy ($`\mathcal{L}_{\mathrm{RCE}}`$) on TVSum, SumMe, and Mr. HiSum.

[A2Summ](https://openaccess.thecvf.com/content/CVPR2023/html/He_Align_and_Attend_Multimodal_Summarization_With_Dual_Contrastive_Losses_CVPR_2023_paper.html) combines multimodal alignment and supervised summarization. Contrastive losses are secondary mechanisms when human importance targets also train the selector.

### SSPVS: pretraining and summarization have different supervision

[Progressive Video Summarization via Multimodal Self-supervised Learning](https://openaccess.thecvf.com/content/WACV2023/html/Li_Progressive_Video_Summarization_via_Multimodal_Self-Supervised_Learning_WACV_2023_paper.html) pretrains video/text representations with coarse correspondence, fine-grained set alignment and masked-frame recovery on YTVT. Its downstream progressive summarizer then minimizes mean-squared error against SumMe/TVSum human frame-importance scores. Classify the complete benchmark route as self-supervised **pretraining plus supervised fine-tuning**, not end-to-end label-free summarization.

The downstream model repeatedly reweights 2-fps GoogLeNet features over one to four stages; optional BERT text enters the first stage. KTS shot means feed a 15%-capacity knapsack. The [official source](https://github.com/HopLee6/SSPVS-PyTorch/tree/e472200069b1697e392f7ac278593c973026b680) exposes another protocol warning: each fold's `test_keys` serve as Lightning validation data, and a post-hoc utility selects the best epoch from those metrics. A clean rerun needs a separate validation partition and fixed epoch rule before final test evaluation.

## Masked and predictive objectives

A **repository-level masked-prediction abstraction** is

```math
\mathcal L_{\mathrm{mask}}=\frac1{|M|}\sum_{t\in M}
\|g_\theta(X_{\bar M})_t-x_t\|_2^2.
```

Mask set $`M`$ defines prediction targets, and $`X_{\bar M}`$ denotes the visible input. Prevent an attention path from seeing the target features through another branch. Reconstruction accuracy measures predictable content; rare salient events can remain poorly modeled. [VideoMAE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html) is an encoder resource, not a standalone video summarizer. SUM-SR's [mask-pretraining ablation](04-reconstruction-generative.md#12-sum-sr-discriminator-free-iterative-reconstruction) is distinct from its main five-iteration configuration.

## Reproduction decisions

Publish positive/negative pair construction, temporal overlap exclusions, augmentation settings, memory bank contents, temperature, selected-set cardinality, and any labeled adaptation set. Avoid claiming a mutual-information guarantee for an arbitrary contrastive loss: the bound depends on sampling assumptions. Evaluate selection quality on held-out videos and retain the complete final decoder.
