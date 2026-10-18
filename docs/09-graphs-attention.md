# Graphs, attention, and Transformers

[Home](../README.md) · [Supervised methods](06-classical-supervised.md) · [Implementations](13-implementations.md)

These mechanisms determine how frames exchange information. Their supervision comes from the objective, not from the architecture name.

## Attention from VASNet to global/local context

[VASNet](https://arxiv.org/abs/1812.01969) is a supervised self-attention summarizer without recurrent sequence decoding. [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM) combines global/local multi-head attention and positional encoding. [GL-RPE](04-reconstruction-generative.md#8-gl-rpe-reconstruction-backbones-with-relative-position) inserts relative position reasoning into unsupervised reconstruction backbones; PGL-SUM and GL-RPE are not the same method.

The **canonical scaled dot-product form**, not an assertion about every method's scaling, is

```math
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad
H=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}+P+M\right)V.
```

$`P`$ represents positional information and $`M`$ is an optional attention mask. A causal $`M`$ prevents future access for streaming. Whole-video attention has quadratic pair storage; a local window or factorization changes the model's receptive field and must be included in cost comparisons.

[CSTA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Son_CSTA_CNN-based_Spatiotemporal_Attention_for_Video_Summarization_CVPR_2024_paper.html) appends a learnable positional token sequence $`E_p\in\mathbb{R}^{T\times D}`$ to the 2-fps 1,024-D GoogLeNet `pool5` sequence $`X\in\mathbb{R}^{T\times D}`$, feeds the 3-channel tensor into `GoogleNet_Attention` (`pool5` before AdaptiveAvgPool2d), applies temporal/dimension softmax maps $`A_T, A_D`$ blended by a learnable ratio $`\epsilon`$, and trains with MSE against `gtscore` (`lr=1e-3`, weight decay `1e-7`, 100 epochs). Its [official source](https://github.com/thswodnjs3/CSTA/tree/b070f7d910b227e040dad66ec9f18b60dac5ea3d) documents two evaluation fixes after publication: SumMe rank correlation previously evaluated against single-frame-rate human summaries instead of 15-frame-sampled summaries (changing SumMe $`\tau/\rho`$ from `0.246/0.274` to `0.205/0.233`), and TVSum rank correlation previously repeated a single split five times (corrected to `0.194/0.255`). Verify the exact evaluation script before comparing published and repository numbers.

## Graph representations

[VideoSAGE, CVPR 2024 workshop](https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/papers/Chaves_VideoSAGE_Video_Summarization_with_Graph_Representation_Learning_CVPRW_2024_paper.pdf) models temporal neighbors as a graph and uses directed/undirected branches for supervised frame selection. Graph edges encode an assumption: temporal proximity, feature similarity, or learned relations produce different neighborhoods.

A **repository abstraction** for message passing is

```math
h_i^{(l+1)}=\sigma\left(W_s h_i^{(l)}+
\sum_{j\in\mathcal N(i)}\alpha_{ij}W_n h_j^{(l)}\right).
```

Do not present it as VideoSAGE's exact EDGE-CONV/SAGE-CONV implementation. A generic graph smoothness penalty is

```math
\mathcal L_{\mathrm{smooth}}=\frac12\sum_{i,j}A_{ij}(s_i-s_j)^2.
```

For nonnegative symmetric $`A`$, this penalizes differences between connected nodes. It can blur brief events; it is an explanatory objective and is not claimed as a term used by every graph summarizer.

## What an architectural comparison must control

Hold features, timestamps, train/validation/test identities, score targets, final shot boundaries and budget solver constant. Count feature extraction and all attention/graph construction costs, not just selector FLOPs. A model that processes pre-extracted tensors is not an end-to-end throughput measurement. Inspect padding masks, positional conventions, graph self-loops, and whether temporal edges cross video boundaries in a batch.
