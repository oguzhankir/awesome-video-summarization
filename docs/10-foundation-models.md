# Foundation models, multimodal fusion, and training-free summarization

[Back to handbook](../README.md) · [Task settings](11-task-settings.md) · [Structured paper records](../data/papers/modern.json)

**Verified: 2026-10-18.** These methods share pretrained representations, but their learning signals differ. Read the supervision column before comparing their results. Official code means author-linked code was found; it does not mean this repository reproduced the model.

## 1. What a foundation model contributes

There are three useful starting points:

| Approach | Where semantics enters | What still needs checking |
|---|---|---|
| Frozen visual/text encoders | Frame, caption, transcript, or query embeddings | Whether the selector is trained against human summaries |
| Caption-and-score pipeline | A video-language model describes scenes; an LLM judges their relevance | Caption omissions, prompt calibration, model version, and final segment selection |
| Instruction-tuned video model | A decoder emits frame indices, text, or both | Summarization fine-tuning data, temporal grounding, and output validation |

A frozen backbone is compatible with a supervised selector. Similarly, no gradient updates at inference do not rule out labeled examples in prompt design. [LLMVS](https://arxiv.org/abs/2504.11199), [V2Xum-LLM](https://arxiv.org/abs/2404.12353), and [CoE](https://arxiv.org/abs/2603.06213) illustrate these separate cases.

The following diagram is an interface blueprint, not the architecture of a particular paper:

```mermaid
flowchart LR
    V["Video and audio"] --> C["Sampled clips and transcripts"]
    C --> E["Pretrained encoders or captioner"]
    Q["Optional query or script"] --> F["Temporal and semantic fusion"]
    E --> F
    F --> S["Importance scores or timestamp proposals"]
    S --> P["Timeline validation and budgeted selection"]
    P --> O["Extractive summary"]
```

For implementations, keep raw timestamps, sampled-frame indices and model tokens in separate fields. A generated frame index refers to the sampled timeline until explicitly mapped back to source time. A text summary needs a separate evaluation record from a selected-video summary.

## 2. Core reading sequence

| Paper | Why read it | Learning signal | Author code |
|---|---|---|---|
| [VideoXum / VTSUM-BLIP](https://arxiv.org/abs/2303.12060) — Jingyang Lin, Hang Hua, Ming Chen, Yikang Li, Jenhao Hsiao, Chiuman Ho, Jiebo Luo; IEEE TMM 2024, accepted 2023 | Joint visual and textual output, shared video representation, and a purpose-built benchmark | Frame BCE and text-generation negative log likelihood | [Official PyTorch](https://github.com/jylins/videoxum) |
| [A2Summ](https://arxiv.org/abs/2303.07284) — Bo He, Jun Wang, Jielin Qiu, Trung Bui, Abhinav Shrivastava, Zhaowen Wang; CVPR 2023 | Timestamp alignment, modality-specific processing, and dual contrastive regularization | Supervised focal classification plus contrastive losses | [Official PyTorch](https://github.com/boheumd/A2Summ) |
| [Scaling Up Video Summarization Pretraining with Large Language Models](https://arxiv.org/abs/2404.03398) — Dawit Mureja Argaw, Seunghyun Yoon, Fabian Caba Heilbron, Hanieh Deilamsalehy, Trung Bui, Zhaowen Wang, Franck Dernoncourt, Joon Son Chung; CVPR 2024 | LLM-generated summary targets and autoregressive visual-summary decoding | Pseudo-summary feature regression; target fine-tuning is a separate setting | Public author release not independently verified |
| [V2Xum-LLM](https://arxiv.org/abs/2404.12353) — Hang Hua, Yunlong Tang, Chenliang Xu, Jiebo Luo; AAAI 2025 | Interleaved frame/timestamp prompts; one language decoder for visual and textual outputs | Supervised instruction tuning, including machine-assisted data | [Official PyTorch](https://github.com/hanghuacs/V2Xum-LLM) |
| [LLMVS](https://arxiv.org/abs/2504.11199) — Min Jung Lee, Dayoung Gong, Minsu Cho; CVPR 2025 | Local LLM reasoning followed by a learned global importance model | Supervised mean squared error | [Official PyTorch](https://github.com/mlee47/LLMVS) |
| [TripleSumm](https://arxiv.org/abs/2603.01169) — Sumin Kim, Hyemin Jeong, Mingu Kang, Yejin Kim, Yoori Oh, Joonseok Lee; ICLR 2026 | Adaptive visual/text/audio fusion and explicit TV versus TVT evaluation | Human or replay-derived score targets | [Official PyTorch](https://github.com/smkim37/TripleSumm) |
| [CLIP-It!](https://proceedings.neurips.cc/paper/2021/hash/7503cfacd12053d309b6bed5c89de212-Abstract.html) — Medhini Narasimhan, Anna Rohrbach, Trevor Darrell; NeurIPS 2021 | Early bridge between generic, free-form query and long-form egocentric summarization | Supervised selector or an ablation without BCE; both use pretrained CLIP/BMT | [Official stub](https://github.com/medhini/clip_it) |
| [Multi-VidSum](https://aclanthology.org/2023.emnlp-main.457/) — Keito Kudo, Haruki Nagasawa, Jun Suzuki, Nobuyuki Shimizu; EMNLP 2023 | Ordered keyframe-caption pairs and task-specific alignment metrics | Supervised joint/iterative models with optional pseudo-pretraining | [Official source](https://github.com/cl-tohoku/Multi-VidSum) |
| [SSPVS](https://openaccess.thecvf.com/content/WACV2023/html/Li_Progressive_Video_Summarization_via_Multimodal_Self-Supervised_Learning_WACV_2023_paper.html) — Haopeng Li, Qiuhong Ke, Mingming Gong, Tom Drummond; WACV 2023 | Multimodal self-supervised representation pretraining plus progressive selection | Self-supervised pretraining, then supervised frame-score MSE | [Official PyTorch](https://github.com/HopLee6/SSPVS-PyTorch) |

### VideoXum: paired outputs change the objective

The VTSUM-BLIP baseline shares video representations between a visual-summary head and a text decoder. Section IV-B uses binary cross-entropy for frame inclusion and autoregressive negative log likelihood for the textual summary. A useful mathematical shorthand is

```math
\mathcal L_{\mathrm{visual}}=\mathrm{BCE}(\hat y,y),\qquad
\mathcal L_{\mathrm{text}}=-\sum_i\log p(w_i\mid w_{<i},V).
```

The paper describes selecting the top 15% of frames. The inspected [F1 helper](https://github.com/jylins/videoxum/blob/11bd4fe3fb51b7dc804dd1519a4914f67bff60df/utils.py#L275-L281) instead includes every score at or above the zero-based threshold `floor(0.15*N)`: it selects `floor(0.15*N)+1` frames when scores are distinct, and potentially more with ties. Preserve this difference between the intended budget and released evaluation behavior. The [evaluation script](https://github.com/jylins/videoxum/blob/11bd4fe3fb51b7dc804dd1519a4914f67bff60df/eval_v2vt_sum.py#L60-L68) computes F1 separately against each human mask, but computes rank correlations against the mean of ten masks. See the [implementation audit](13-implementations.md#multimodal-and-query-focused-code) before comparing this frame protocol with KTS keyshot evaluation.

The dataset partition is 8,000 training, 2,001 validation and 4,000 test videos. Use the archival **2024** journal year; the repository's TMM 2023 label reflects its earlier acceptance timeline. [Paper, Sections III-A and IV](https://arxiv.org/html/2303.12060v3) · [Official project and model zoo](https://videoxum.github.io/).

### A2Summ: contrastive is not automatically self-supervised

A2Summ aligns transcript sentences with the frames in their time intervals, then uses alignment-guided self-attention and modality experts. Its classification targets are annotated keyframes/key sentences, and its intra-sample contrastive positives also depend on ground-truth summaries. Thus the complete model belongs in supervised comparisons. Contrastive regularization describes its mechanism, not the absence of labels. The author project provides a [presentation, slides and poster](https://boheumd.github.io/A2Summ/); the full objective is in [Section 3.4](https://arxiv.org/html/2303.07284v3).

### LfVS: distinguish pseudo-summary prediction from input reconstruction

Argaw and colleagues retain timestamps while asking an LLM to summarize transcripts, then map the selected sentences back to video. A multimodal encoder-decoder learns to predict those **summary** features; Eq. (10) minimizes squared error including the end-of-sequence token. At inference, nearest-neighbor retrieval maps decoded representations to source moments. This is externally supervised target construction, even though annotation is automated. It is not the same objective as reconstructing the complete source video from a bottleneck. [Primary method, Sections 3–4](https://arxiv.org/html/2404.03398v1).

### LLMVS and V2Xum-LLM: two routes from language to selection

LLMVS uses LLaVA captions and Llama-2 local context embeddings, followed by a self-attention aggregator trained against human importance scores. V2Xum-LLM trains its adapter and decoder to emit temporal indices and/or natural language. Their supervised objectives are, respectively, frame-score MSE and output-token negative log likelihood. Freeze status of the captioner or visual encoder does not change these selector-level categories. The pinned LLMVS source loads visual tensors but its model path uses only two LLaMA embedding streams; it also selects checkpoints on the test-named partition and evaluates that partition again. [LLMVS, Sections 3.3–3.5](https://arxiv.org/html/2504.11199v2) · [V2Xum-LLM, Methodology and Training](https://arxiv.org/html/2404.12353v3).

## 3. TripleSumm: exact identity and protocol audit

The verified title is **TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization**, by **Sumin Kim, Hyemin Jeong, Mingu Kang, Yejin Kim, Yoori Oh, and Joonseok Lee**, published at **ICLR 2026**. The arXiv identifier is **2603.01169**. Exact-name searches for “Triple-Sum”, “Triple Sum” and “Triplet-Sum” did not establish a separate video-summarization paper. Preserve the verified spelling; do not create aliases as separate citations. [Paper](https://arxiv.org/abs/2603.01169) · [ICLR paper](https://openreview.net/pdf?id=x74NsHGywD).

```mermaid
flowchart TD
    V["Visual embeddings"] --> P["Modality projections"]
    T["Transcript or caption embeddings"] --> P
    A["Audio embeddings"] --> P
    P --> F["Per-frame fusion tokens"]
    F --> M["Multiscale temporal refinement"]
    M --> C["Cross-modal refinement"]
    C --> H["Importance head"]
    H --> R["Score regression"]
    H --> K["KTS, mean shot scores, 0/1 knapsack"]
```

The main method alternates temporal attention within modalities and fusion across modalities. Its training objective is

```math
\mathcal L(S,\widehat S)=\lVert S-\widehat S\rVert_2^2
```

against importance targets. It is **supervised** on human-annotated datasets and uses behavioral weak supervision on MoSu. MoSu pretraining followed by target fine-tuning is distinct from direct transfer. The main results separate traditional TV folds from TVT folds with dedicated validation and test partitions. [Sections 3.3 and 5.1; Appendix B.5](https://arxiv.org/html/2603.01169v1).

The [author project](https://sumin-kim.com/TripleSumm-page/) introduces MoSu, a 52,678-video dataset with synchronized visual, text and audio features, and interactive examples of modality contributions. Replay statistics measure viewer behavior; they should not be described as independent human judgments of summary quality. The [code repository](https://github.com/smkim37/TripleSumm) links pinned datasets and MoSu/Mr. HiSum checkpoints, with Python 3.10, PyTorch 2.5.1 and CUDA 12.1 instructions. Its main evaluator reports raw-score correlations and fixed-five-second-shot mAP; the shipped 15%-knapsack decoder is not invoked by the main path. These artifacts were inspected, not executed here.

Three reporting traps require particular care:

- **Table 3 “Ours (MoSu)” includes target fine-tuning.** Only the separate Table 5 long-video experiment is direct MoSu-to-new-video transfer. A zero-shot transfer label does not imply that the selector was never trained.
- **Main Table 2 and Appendix D prose disagree on the MoSu rank values.** Appendix Table XI agrees with the main table. Preserve the table identifier beside any copied number; do not silently select the higher prose claim. Correlations, highlight mAP and final keyshot overlap are separate evaluations. [Primary tables and appendices](https://arxiv.org/html/2603.01169v1).
- **External-dataset captioner names differ.** Section 5.1 names `Qwen2.5-VL-7B-Instruct`, while Appendix B.4 links `Qwen/Qwen2-VL-7B-Instruct`. Resolve the actual caption cache and model revision before reproducing these experiments. [Primary preprocessing descriptions](https://arxiv.org/html/2603.01169v1).

## 4. The 2026 audit window: similar labels, different tasks

| Work | Output and supervision | Protocol boundary |
|---|---|---|
| [TRINITY](https://arxiv.org/abs/2608.29577v1) | Supervised perspective-specific Event/Emotion/Nature segment scores | “Personal-style” is three constructed views, not an individual user model; paper, README and manifests conflict on counts |
| [SGWIB](https://arxiv.org/abs/2609.13966v1) | Replay-supervised visual or audio highlight ranking, trained separately | No audiovisual fusion; MoSu Table III and prose values conflict, so no row is admitted |
| [Zero-shot text/synthetic-image highlights](https://arxiv.org/abs/2609.14790v1) | Training-free metadata-conditioned frame ranking | TVSum's best uses text+image+style; SumMe's best is text-only with video-derived category bootstrapping |
| [KnowVis](https://arxiv.org/abs/2609.03742v2) | Training-free generated pedagogical image from lecture transcripts/slides | Not a temporal skim; proprietary models, no executable code, one unsplit generated-output dataset |
| [Unified Agentic Video Editing](https://arxiv.org/abs/2609.12769v1) | Training-free 90–110-second audiovisual narrative from private metadata | Five episodes; Gemini-2.5-Flash answers a fixed comprehension quiz, and editing agents do not inspect pixels |
| [Semantic Action Graph](https://arxiv.org/abs/2609.20768v1) | Five personalized sports clips plus narration and graph UI | One match and privileged event feed; no standard summary metric or code |
| [Vision-language hierarchical model](https://doi.org/10.1002/ail2.70039) | Supervised CLIP/BLIP-2 importance model and KTS skim | 20% budget, mean-all-annotators F1 and GoogLeNet KTS boundaries isolate its headline rows |
| [Audio sports highlights](https://arxiv.org/abs/2609.17923v1) | Supervised 2-second sports clip ranking with audio/visual/fused GRUs | Paper-local three-seed splits of 317 videos have no public identities |
| [TSMD](https://arxiv.org/abs/2609.39051v1) | Weakly supervised multimodal highlight scoring with temporal and full-stream dropout | Zero-masking tests robustness on MoSu/Mr. HiSum; segment mAP is not final keyshot F1 |
| [EviDETR](https://arxiv.org/abs/2609.30724v1) | Query-conditioned moment retrieval and highlight scores with evidence transfer | QVHighlights validation HD-mAP/HIT@1 are adjacent task metrics, not generic summary quality |
| [SD-MVSum](https://arxiv.org/abs/2610.03000v1) | Supervised/weakly supervised multi-modal script-driven frame scoring | Evaluates both step-by-step and free-form scripts on SM-VideoXum (BCE, top-15% frames) and SM-Mr. HiSum (MSE, 15%-knapsack F1 and $`\tau/\rho`$) |
| [TRIM](https://arxiv.org/abs/2610.04462v1) | Training-free VLM prompt construction, CLIP cosine scoring, and two-stage filtering | Evaluates full 25-video SumMe and 50-video TVSum sets alongside five 20%-test splits; full-dataset and split averages must stay separate |
| [TRIMMER](https://arxiv.org/abs/2610.06423v1) | Training-free/zero-shot VLM pseudo-summary distillation into a small per-video MLP | Fits a two-layer MLP per test video for 50 epochs against VLM pseudo-summary targets; zero human labels, not zero gradient steps |
| [VASTSum](https://arxiv.org/abs/2610.04930v1) | Self-supervised alignment-consistency pretraining plus supervised MSE fine-tuning | Reports both TVT and 5FCV rank correlations on TVSum/SumMe and 5-second-shot mAP on Mr. HiSum; no public code or split manifest yet |
| [LVSum](https://arxiv.org/abs/2610.06808v1) | Long-form (15–60 min) human-annotated scene-importance benchmark and MLLM evaluation | 82 videos, 3,253 scenes, 10% key-scene knapsack budget; shows sharp MLLM performance drop beyond 30 minutes |
| [ClipSum](https://arxiv.org/abs/2610.14628v1) | Supervised dual-branch multi-stage visual-semantic fusion (I3D + CLIP + BLIP-2/LLaMA-3) | Reports SumMe/TVSum canonical/augmented/transfer F1 and $`\tau/\rho`$, plus QFVS F1; repository is currently a stub |
| [TR-EduVSum](https://arxiv.org/abs/2610.12573v1) | Supervised bilingual (English/Turkish) educational lecture video/text summarizer | Ground-truth keyshots and summaries are LLM-synthesized rather than multi-annotator human labels |
| [HAS](https://arxiv.org/abs/2610.16035v1) | Training-free Hierarchical Adaptive Sampling for MLLM video understanding | Selects a token-budgeted keyframe sequence ($`N_s=32/64`$) via shot allocation and submodular utility for Video-MME/LongVideoBench/MLVU |
| [AdaFrame](https://arxiv.org/abs/2610.19127v1) | Training-free 3-tier deduplication and SVD information-budgeted keyframe selection | Evaluated via downstream VLM ad classification accuracy and API cost on Pitt Ads rather than human keyshot F1 |
| [FramePack-U](https://arxiv.org/abs/2610.18120v1) | Training-free uncertainty-guided multi-level keyframe packing for Video-LLMs | Compresses context tokens inside LLaVA-Video-7B-Qwen2 / InternVL2.5-8B for long-video QA rather than keyshot F1 |
| [PIGEONS](https://arxiv.org/abs/2610.17538v1) | Unsupervised dual-prompting (visual + textual) video-to-text summarization via pseudo-triplets | Evaluates natural-language summaries on Shot2Story, VideoXum (text), and ActivityNet without ground-truth video-summary training pairs |

TRINITY's released source is substantive, but paper weight decay and epoch settings differ from defaults, checkpoints are absent, and advertised baseline signatures do not match the combined solver. KnowVis's official repository contains only a README and two images. SD-MVSum ships complete training/inference code and Zenodo HDF5/checkpoint archives with MD5 hashes, whereas TRIM, ClipSum, and QEVA currently expose placeholder stub repositories. These are different reproducibility states even though all have author-linked repositories. See the [pinned resource records](generated/resources.md).

The metadata-conditioned zero-shot paper reports its strongest TVSum correlations with text and synthetic-image descriptors plus reference-frame style transfer, but its strongest SumMe correlations use text descriptors alone. SumMe first infers a coarse category with Qwen2.5-VL, so this is not the same native-metadata setting as TVSum. Sampling cadence, prompt/model snapshots, seeds and evaluated-video manifests are absent; the four recorded correlations remain isolated.

TSMD studies synthetic feature loss in behavior-supervised highlight detection: temporal masks remove 50% of feature positions, while stream masks remove visual, audio or text features. Its 16 admitted Table 1 mAP@15 rows preserve the retrained baseline, three dropout policies, two datasets and both failure regimes; the values remain author-reported and split-isolated. EviDETR combines query-conditioned temporal grounding with clip highlight prediction on QVHighlights. Its HD-mAP and HIT@1 rows are listed separately from the generic keyshot catalog.

## 5. Training-free pipelines and the data used to design them

| Work | Actual output and inference | Qualification needed |
|---|---|---|
| [Prompts to Summaries](https://arxiv.org/abs/2506.10807), Mario Barbara and Alaa Maalouf; arXiv 2025, revised 2026 | Scene captions and prompted judgments become frame scores; optional natural-language query | No selector gradient training, but SumMe and TVSum use different normalization selected through dataset experiments |
| [Context-Aware Pseudo-Label Scoring](https://arxiv.org/abs/2510.17501), Yuanli Wu, Long Zhang, Yue Du and Bin Li; arXiv 2025 | Captions and contextual rubric prompts produce importance scores | Human-annotated examples help construct the rubric; label-assisted calibration, despite the zero-shot title |
| [Cut to the Chase / CoE](https://arxiv.org/abs/2603.06213), Xiaoxing You, Qiang Huang, Lingyu Li, Xiaojun Chang and Jun Yu; CVPR 2026 | Event graph, visual grounding and event evolution produce a textual summary | No gradient training; five training-set summaries serve as style references |
| [KnowVis](https://arxiv.org/abs/2609.03742v2), Yi Xu, Yifan Hou and Xiaoyu Zhang; arXiv 2026 | Knowledge graph and lecture evidence become a pedagogical visual image | Proprietary generation/judging APIs; generated dataset outputs are not human references |
| [Zero-shot text/synthetic-image highlights](https://arxiv.org/abs/2609.14790v1), Michal Byra, Alberto Presta, Grzegorz Stefanski and Krzysztof Arendt; arXiv 2026 | Metadata prompts become CLIP text/image prototypes and frame rankings | Metadata-dependent; SumMe category is inferred from the video; no summary decoder/code |
| [TRIM](https://arxiv.org/abs/2610.04462v1), Evelynvensia, Jia-Hong Huang, Hsiao-Chien Chang, Hong-Han Shuai, Wen-Huang Cheng, Jun-Cheng Chen; arXiv 2026 | LLaVA-NeXT-Video generates global/local descriptions; CLIP cosine similarity plus Adaptive K-Means and distribution pruning produce frame scores | Evaluates both 5-fold 20% test splits and all 25/50 videos; check which evaluation mode a table reports |
| [TRIMMER](https://arxiv.org/abs/2610.06423v1), Kangsan Kim, Geonho Lee, Myungjoo Kang; arXiv 2026 | VLM generates binary shot pseudo-summaries via divide-and-conquer + CoT; a 2-layer MLP is trained per video on those pseudo-labels | Requires zero human labels, but runs 50 test-time gradient steps per video (`lr=1e-4`) |

**Prompts to Summaries.** Sections 3.3–3.4 define scene judgment, score normalization, temporal smoothing, and frame consistency/uniqueness weighting. The standard-video experiments use a 15% duration budget, but VidSum-Reason uses 36%; do not merge these results. Its [official implementation](https://github.com/mario998-hash/ZeroShotVideoSummary/tree/405acfe8dc5cbf4aa9ba30e6d711a4fc25d33ad9) requires a video-language model and an OpenAI API key. The README retains an older clone URL, so use the paper-linked repository and check configuration against the inspected revision. [Primary methodology and protocol](https://arxiv.org/html/2506.10807v3).

**Context-Aware Pseudo-Label Scoring.** Section 3 explicitly starts with a small human-annotated subset and creates dataset-specific rubrics before inference. A clean reproduction needs the identities of those examples and evidence that evaluation videos were excluded from rubric design. The current manuscript also contains placeholder venue metadata and inconsistent baseline attributions. Its [author-linked code](https://github.com/wuyuanli60-svg/Context-Aware-Pseudo-Label-Scoring-for-Zero-Shot-Video-Summarization/tree/03ba697264aff14c0af7e0340b15034a77293f9e) is useful for inspecting prompts, but no headline numerical comparison is accepted here. [Primary paper, Sections 3–4](https://arxiv.org/html/2510.17501v3).

**CoE.** Section 3 structures summarization through a hierarchical event graph, cross-modal grounding, event-evolution reasoning and domain-adaptive generation. Section 4.1 selects five reference summaries from training data for style adaptation. Its output is text; entity F1, ROUGE and BERTScore do not measure temporal overlap of selected keyshots. The [official code](https://github.com/youxiaoxing/CoE) needs dataset preparation, MongoDB and model-serving configuration. Its documented pip command targets a conda-export file, several evaluator dependencies are absent, and the committed main call disables question/BERTScore refinement. [Primary paper](https://arxiv.org/html/2603.06213v1).

## 6. Reproduction and open questions

For each foundation-model run, record model/checkpoint identifiers, prompt text, sampling timeline, decoding parameters, example provenance, caption/transcript cache hashes, API usage and cost, and the final selection procedure. Measure feature extraction and captioning separately from selector latency. Model availability and benchmark contamination remain unverified unless independently audited.

The most useful open questions are testable: does direct visual evidence rescue events lost by the captioner; do normalization choices survive a held-out dataset; does adaptive audio fusion help beyond speech; and do semantically plausible textual summaries preserve causal order and exact timing? The current ledger provides representative starting points, not an exhaustive 2026 survey or independently reproduced leaderboard.
