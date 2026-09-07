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

[CSTA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Son_CSTA_CNN-based_Spatiotemporal_Attention_for_Video_Summarization_CVPR_2024_paper.html) uses 2D convolution over image-like frame-feature representations to capture spatiotemporal relationships. Its [official source](https://github.com/thswodnjs3/CSTA) documents evaluation corrections. Verify the timeline implementation before treating a published number as identical to a current-code number.

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
