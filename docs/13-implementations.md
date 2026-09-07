# Implementations, artifacts, and reproduction routes

**Source audit: 2026-09-08.** Start with the [learning path](16-learning-path.md) and the local [summary-construction example](../examples/summary_baselines.py), then choose an upstream repository below. The machine-readable [resource registry](../data/resources.json) records attribution, source commits, environment information, licenses, artifact status and limitations.

This audit read GitHub metadata, READMEs, dependency files and selected training, preprocessing and evaluation source. It did **not** install upstream environments, download model/data archive bodies, run training, or reproduce reported scores. “Official” identifies a release by the method authors; it does not establish successful execution, correct evaluation, or official dataset provenance.

## 1. Pick a starting point

| Your next step | Start here | What to produce |
|---|---|---|
| Understand scores, segments, budgets and F1 | [Local example](../examples/summary_baselines.py), [evaluation chapter](02-evaluation.md) | Compare constant, random and uniform baselines; explain mean versus duration-weighted shot values |
| Inspect benchmark input files | [h5py guide](https://docs.h5py.org/en/stable/quick.html), [dataset registry](03-datasets.md) | A manifest of feature shapes, timeline indices, annotations and split IDs |
| Understand policy-gradient selection | [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) | Trace input → probability → sampled actions → diversity/coverage reward → update |
| Understand reconstruction training | [SUM-GAN-AAE](https://github.com/e-apostolidis/SUM-GAN-AAE) | Trace each network's loss and compare soft weighted features with a hard summary |
| Study attention without recurrent generators | [VASNet](https://github.com/ok1zjf/VASNet), [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM) | A supervised comparator with a declared checkpoint rule |
| Follow a raw-video preprocessing path | [DSNet](https://github.com/li-plus/DSNet) | Decode one permitted video, inspect feature/timestamp alignment, then generate a skim |
| Study visual and textual summaries together | [VideoXum](https://github.com/jylins/videoxum) | Keep visual F1, rank correlation, caption metrics and cross-modal alignment separate |
| Study query-focused summaries | [UniVTG](https://github.com/showlab/UniVTG) | Evaluate semantic shot matching using the QFVS-specific inputs and budget |
| Audit why a metric changes | [Rethinking EVS](https://github.com/mayu-ot/rethinking-evs), [original TVSum evaluator](https://github.com/yalesong/tvsum) | Hold scores fixed while varying segmentation and reference construction |

These are routes through available source, not claims that the legacy packages run on a current laptop unchanged. Supervised comparators remain useful for learning architecture and evaluation but do not enter the unsupervised result pool.

## 2. Method implementation registry

Commit links identify the exact source inspected. Full 40-character identifiers are stored in [resources.json](../data/resources.json).

| Implementation and attribution | Source snapshot | Environment stated by authors | Artifacts observed | Main reproduction caveat |
|---|---|---|---|---|
| **vsLSTM / dppLSTM**, official, supervised | [0ee0a094](https://github.com/kezhang-cs/Video-Summarization-with-LSTM/tree/0ee0a0948872544567ecede76868287042470f75) | Python 2.7+, Theano 0.7+, MATLAB | Two trained model files in tree; data-download instruction incomplete | Old environment; earlier HDF5 layout; released TVSum reference budget differs from paper |
| **DR-DSN**, official, unsupervised | [041606fe](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce/tree/041606fec8e2ba7f4c4900064a76d965a473e53c) | Python 2.7, PyTorch 0.4.0 | README links alternate dataset archive; no pretrained weights advertised | Inaccessible original QMUL host; features and video rendering require explicit preparation |
| **SUM-GAN**, community implementation by Jaemin Cho | [fb6d5bf7](https://github.com/j-min/Adversarial_Video_Summary/tree/fb6d5bf70479373f96f2d944c672af8286c9bc89) | PyTorch / torchvision, versions unpinned | Feature-extraction and training source; no checkpoint found | Changes backbone and adversarial training; no claim of original-paper reproduction |
| **SUM-GAN-AAE**, official, unsupervised training | [3bced18d](https://github.com/e-apostolidis/SUM-GAN-AAE/tree/3bced18d30f04e74e2189601b30ad5d170758250) | Python 3.6, PyTorch 1.0.1 | GoogLeNet HDF5 `.rar` archives and five split JSONs in tree | Separate multi-user/synthesized-reference scripts; epoch choice must be stated |
| **VASNet**, official, supervised | [c3787531](https://github.com/ok1zjf/VASNet/tree/c3787531486f74789dc5e92758edf51e24f56e6d) | Python 3.5.2, PyTorch 0.4.1, OR-Tools 6.9.5824 | Both historical Box dataset/model URLs confirmed HTTP 404 on 2026-09-08 | Selects best epoch on `test_keys` F1; no replacement checkpoint verified |
| **PGL-SUM**, official, supervised | [81d0d6d0](https://github.com/e-apostolidis/PGL-SUM/tree/81d0d6d0ee0470775ad759087deebbce1ceffec3) | Python 3.8.8, PyTorch 1.7.1; TensorFlow log tooling | HDF5 `.rar` archives; author-linked Zenodo model archive | Manual inference paths; training-loss-based checkpoint selection must be preserved |
| **DSNet**, official, supervised | [1804176e](https://github.com/li-plus/DSNet/tree/1804176e2e8b57846beb063667448982273fca89) | Python 3.6, PyTorch 1.1.0, torchvision 0.3.0; graph extensions | Dataset and two pretrained-model archive links; custom-video tools | Default sampling is every 15 decoded frames; checkpoint selection reads `test_keys` |
| **VideoXum / VTSUM-BLIP**, official, supervised | [11bd4fe3](https://github.com/jylins/videoxum/tree/11bd4fe3fb51b7dc804dd1519a4914f67bff60df) | Python 3.8, PyTorch 1.10.1, CUDA 11.1, transformers 4.15.0 | Author HF feature, annotation and weight listings inspected | Loader length handling and F1 threshold have material protocol consequences |
| **UniVTG**, official for UniVTG; derived QFVS tooling | [32659ac7](https://github.com/showlab/UniVTG/tree/32659ac7aeba21742a63274f30eba785fc57e247) | Python 3.8, PyTorch 2.0.1 in requirements | Author Drive links for processed QFVS artifacts and models | Slurm/CUDA assumptions; QFVS semantic F1 differs from generic temporal F1 |
| **FCSNA-QFVS**, author notebook, supervised | [3832ec78](https://github.com/srkds/Query-Focused-Long-Video-Summarization/tree/3832ec786e59a2bd2e9da997a63b03bc1447988e) | PyTorch/Jupyter, versions unpinned | Training and qualitative notebooks | Author says release needs work; Colab paths, fixed shapes and CUDA operations |

**Code-search gaps.** No official Cycle-SUM implementation was verified after searches for the exact method name, its authors and GitHub code, followed by the [primary paper](https://arxiv.org/abs/1904.08265). This is “not found in this audit,” not proof that no source exists. The [SUM-GAN primary paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.html) was not linked to a verified author implementation; the explicitly unofficial project above should retain that label. For the original QFVS memory-network method, the [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Sharghi_Query-Focused_Video_Summarization_CVPR_2017_paper.pdf) points to an author project for data/evaluation code; that page could not be fetched in this audit. Later UniVTG and FCSNA releases do not establish release of that original model.

## 3. Read these source paths before training

### Historical recurrent and generative code

The official [dppLSTM data loader](https://github.com/kezhang-cs/Video-Summarization-with-LSTM/blob/0ee0a0948872544567ecede76868287042470f75/codes/tools/data_loader.py) predates the common per-video HDF5 schema. Its README describes `idx`, `fea_i`, `gt_1_i`, and `gt_2_i`. An archive of `video_i/features` groups is not a drop-in substitute. The [TVSum evaluator](https://github.com/kezhang-cs/Video-Summarization-with-LSTM/blob/0ee0a0948872544567ecede76868287042470f75/codes/evalTVSum/evaluate_TVSum.m) sets its reference budget to `0.18`; preserve and disclose this discrepancy when reconstructing published code behavior.

The community SUM-GAN [README](https://github.com/j-min/Adversarial_Video_Summary/blob/fb6d5bf70479373f96f2d944c672af8286c9bc89/README.md) documents ResNet-101 features and altered discriminator updates. Its [extractor](https://github.com/j-min/Adversarial_Video_Summary/blob/fb6d5bf70479373f96f2d944c672af8286c9bc89/feature_extraction.py) also supports ResNet-152. This is useful implementation material, but these changes affect distances, reconstruction scale and likely segmentation.

SUM-GAN-AAE provides distinct [multi-user evaluation](https://github.com/e-apostolidis/SUM-GAN-AAE/blob/3bced18d30f04e74e2189601b30ad5d170758250/evaluation/check_fscores_summe.py) and [synthesized-reference evaluation](https://github.com/e-apostolidis/SUM-GAN-AAE/blob/3bced18d30f04e74e2189601b30ad5d170758250/evaluation/check_fscores_summe_with_gts.py). The former emits test scores for each epoch; source inspection alone does not justify claiming a label-free checkpoint rule. State which epoch was selected and why.

### Policy-gradient and supervised comparisons

DR-DSN's [main.py](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce/blob/041606fec8e2ba7f4c4900064a76d965a473e53c/main.py) optimizes sampled-action rewards and saves the final configured epoch. In that inspected path, `gtscore` is read when exporting results, not as a training loss. Trace [rewards.py](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce/blob/041606fec8e2ba7f4c4900064a76d965a473e53c/rewards.py) to understand its empty-selection and temporal-distance handling. Its split generator explicitly makes repeated random partitions; “five splits” does not imply five disjoint test folds.

VASNet's model source is [`vasnet_model.py`](https://github.com/ok1zjf/VASNet/blob/c3787531486f74789dc5e92758edf51e24f56e6d/vasnet_model.py), as confirmed in the inspected repository tree. Its [training code](https://github.com/ok1zjf/VASNet/blob/c3787531486f74789dc5e92758edf51e24f56e6d/main.py) evaluates `self.test_keys` every epoch and keeps the largest F1. That set therefore also influences checkpoint choice. PGL-SUM's [selection script](https://github.com/e-apostolidis/PGL-SUM/blob/81d0d6d0ee0470775ad759087deebbce1ceffec3/evaluation/choose_best_epoch.py) instead examines changes in training loss. These are different experimental procedures even when features and split names match.

DSNet's [video helper](https://github.com/li-plus/DSNet/blob/1804176e2e8b57846beb063667448982273fca89/src/helpers/video_helper.py) extracts normalized 1,024-dimensional GoogLeNet features every `sample_rate` decoded frames and runs KTS on that sequence. The [custom-data script](https://github.com/li-plus/DSNet/blob/1804176e2e8b57846beb063667448982273fca89/src/make_dataset.py) defaults to 15: this equals 2 fps only on 30-fps video. Its [anchor-based trainer](https://github.com/li-plus/DSNet/blob/1804176e2e8b57846beb063667448982273fca89/src/anchor_based/train.py#L41) builds the validation loader from `split['test_keys']` and saves the checkpoint with the highest F1 on that loader. Those test-named examples therefore influence checkpoint selection in the inspected release. Whether a particular published number used this exact code and split remains independently unverified; a separate validation partition is required for an independent held-out test protocol.

### Multimodal and query-focused code

VideoXum's [loader](https://github.com/jylins/videoxum/blob/11bd4fe3fb51b7dc804dd1519a4914f67bff60df/data/activitynet_dataset.py) reads BLIP `.npz` key `features`, and VT-CLIP keys `vision` and `text`. It uses a maximum length of 512 by default and can shorten mismatched feature/reference sequences. Log lengths before and after these operations.

The released [F1 helper](https://github.com/jylins/videoxum/blob/11bd4fe3fb51b7dc804dd1519a4914f67bff60df/utils.py) uses the zero-based threshold at index `floor(0.15*N)` and retains every score at least that threshold. With distinct scores this selects `floor(0.15*N)+1` frames; ties can admit still more. This is not a strict 15% top-k budget. The [evaluation script](https://github.com/jylins/videoxum/blob/11bd4fe3fb51b7dc804dd1519a4914f67bff60df/eval_v2vt_sum.py) computes correlation against the mean of ten human masks, while its F1 retains mean/max reference aggregation. Its VT-CLIPScore path uses another top-k selection. Report the actual metric path, not a generic “top 15%” label.

UniVTG's [QFVS evaluation](https://github.com/showlab/UniVTG/blob/32659ac7aeba21742a63274f30eba785fc57e247/eval/qfvs.py) matches semantic shot tags through maximum-weight bipartite matching. Its [inference path](https://github.com/showlab/UniVTG/blob/32659ac7aeba21742a63274f30eba785fc57e247/main/inference_qfvs.py) reads segmented HDF5 features, uses a configurable `top_percent`, explicitly allocates CUDA tensors and trims a noted video4 feature/tag mismatch. Its TVSum highlight-detection path belongs to a different metric/task protocol. The FCSNA-QFVS [notebook](https://github.com/srkds/Query-Focused-Long-Video-Summarization/blob/3832ec786e59a2bd2e9da997a63b03bc1447988e/FCSNA-QFVS.ipynb) is useful for inspection, but hard-coded `/content/drive/MyDrive/Research` paths, 20×200 layouts and query-tag targets require adaptation.

## 4. Dataset and evaluation tools

| Tool | Inspected snapshot | Useful function | Constraint |
|---|---|---|---|
| [TVSum original release](https://github.com/yalesong/tvsum) | [7bf3fb8c](https://github.com/yalesong/tvsum/tree/7bf3fb8ca032f5f4b42e22ba41f5e2490b819f96) | MATLAB annotation handling and original reference construction | Fixed 60-frame reference units differ from later KTS-derived references |
| [Rethinking EVS](https://github.com/mayu-ot/rethinking-evs) | [7a3b05e6](https://github.com/mayu-ot/rethinking-evs/tree/7a3b05e63ba531c89ac022e3a98bfd37d22d60a2) | Random-summary baseline analysis and rank-statistics notebooks | Python 3.7.3/SciPy 1.2.1 environment; original KTS artifact marked unavailable by authors |
| [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html) | Live official documentation | Timestamp-aware sampling, trimming and concatenation | Keep decoded-frame and timestamp mappings |
| [h5py quick start](https://docs.h5py.org/en/stable/quick.html) | Live official documentation | Inspect groups, tensor shapes and dtypes | A compatible shape does not prove matching features |
| [CLIP reference code](https://github.com/openai/CLIP) | README/documentation inspected; no commit pinned | Learn author preprocessing and paired image/text embeddings | Supplies a representation, not an end-to-end summarization benchmark |

## 5. Artifact verification and licensing

Observed artifact stages are deliberately separate:

| Stage | What this audit established |
|---|---|
| Source accessible | Selected repository text files were fetched and read at the listed commit |
| Artifact advertised | Source README/tree or provider listing names a feature archive/checkpoint |
| Binary downloaded and inspected | **Not performed for upstream model/data archives in this code audit** |
| Runtime verified | **Not performed for upstream projects** |
| Paper result reproduced | **Not performed** |

VideoXum's author [feature listing](https://huggingface.co/datasets/jylins/videoxum/tree/main) exposes BLIP and VT-CLIP feature archives; its [model listing](https://huggingface.co/jylins/vtsum_blip/tree/main) exposes `vt_clip.pth`, `vtsum_tt.pth` and `vtsum_tt_ca.pth`. PGL-SUM advertises [Zenodo models](https://doi.org/10.5281/zenodo.5635735), but that record returned HTTP 429 to this audit's browser. That is a rate-limit observation, not a dead-link finding.

VASNet's historical [dataset archive](https://kingston.box.com/shared/static/zefb0i17mx3uvspgx70hovt9rr2qnv7y.zip) and [pretrained-model archive](https://kingston.box.com/shared/static/2u2nqelxsptzefp1mno4z28kimmbljw2.zip) both returned HTTP 404 in the external audit on **2026-09-08**. These links remain recorded for provenance; no replacement weights were verified.

SUM-GAN-AAE and PGL-SUM have academic non-commercial code licenses. The original dppLSTM notice says CC-BY without a version and restricts its metadata to research purposes. Several inspected trees, including the community SUM-GAN and VideoXum root, lack a repository-wide license file; a model-card license or vendored dependency notice does not automatically license the whole tree. Consult each resource's recorded notice before reusing code. This handbook's MIT license does not relicense external artifacts.

## 6. A reproducible first run

1. Record the selected source commit and required framework versions before adapting code.
2. Obtain data/features through the stated owner; record file checksum, size and access date after download.
3. Inspect one video's feature shape, original length, `picks`, boundaries and reference count. Distinguish dataset-author annotations from researcher-produced feature archives.
4. Freeze train/validation/test IDs, random seeds and checkpoint selection. Keep final-test labels out of model selection.
5. Run constant/random/uniform baselines through the **same** reference conversion, segmentation, pooling and budget code.
6. Run one inference case; inspect the resulting timeline and play the skim before scaling up.
7. Report metrics with the full protocol tuple and label any code changes relative to the source snapshot.

Source availability is the beginning of a reproduction. Keep intermediate failures and corrections in the experiment record so another learner can follow the same path.
