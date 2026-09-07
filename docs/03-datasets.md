# Dataset and Pre-extracted Feature Registry

## 1. Provenance rules

“Official” means released by the dataset or benchmark authors. A feature package produced by a later method's authors is a **paper-author derivative**, even when it is widely used. Extraction code is not evidence that ready-made tensors were released.

Dataset/source pages were reviewed on **2026-09-08 UTC**. The structured [dataset registry](../data/datasets.json) separates task, annotation geometry, access, license, features, and unresolved fields. `verified_on` records a source review, not a successful download of every archive.

Exact archive measurements and checksums inherited from the **2026-09-06** handbook are explicitly labelled below. They were **not recomputed in this run** and must be checked against the downloaded bytes before an experiment. A working project page does not prove that its linked media archive remains downloadable.

YouTube-based datasets are vulnerable to deletion, geographic restrictions, and content replacement. This registry distinguishes hosted video bytes, URL lists, annotation-only packages, and derived features.

## 2. Core and adjacent datasets

| Dataset | Corpus and raw duration | Annotation unit and density | Canonical evaluation and splits | Primary project and download |
|---|---|---|---|---|
| **SumMe** | **25** mostly raw or weakly edited YouTube user videos covering holidays, events, and sports; approximately **1–6 min/video**. The 25 durations in the paper's Table 1 sum to **3,999 s = 66 min 39 s** (mean **159.96 s**); this is an inherited table calculation, not a paper-reported aggregate or a measurement repeated here. | **15–18 users/video**, **390 reference summaries total**. Each user watched the complete video and selected temporal intervals; summaries were constrained to **5–15%** of source duration. Annotation omitted audio. | Temporal-overlap precision/recall/F1 with a usual **≤15%** machine-summary budget. The original protocol averages a machine summary's F1 across users; later learning benchmarks commonly use the maximum user F1, so results must name the aggregation. No original train/test split. Later work often uses random 80/20 partitions packaged as five splits; fully unsupervised work often evaluates all 25. | [ECCV 2014 paper](https://link.springer.com/chapter/10.1007/978-3-319-10584-0_33), [annotation supplement](https://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/ECCV_2014/html/8695/86950505/esm2.pdf), [CVF dataset index](https://cove.thecvf.com/datasets/615), [institutional ZIP](https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip). The 390 count appears in the authors' [CVPR 2015 follow-up](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gygli_Video_Summarization_by_2015_CVPR_paper.pdf). |
| **TVSum / TVSum50** | **50** Creative-Commons YouTube videos; **10** TRECVid MED categories and **5/category**; collection range **2–10 min/video**; inherited release statistics: **3 h 29 m 42 s**, **352,356 frames** (not recomputed). | Every uniform **2-second shot** received a **1–5 importance rating** from **20 workers/video**: **1,000 complete video-level responses**. Workers saw the title and watched the full video with audio muted. | The official evaluator decodes **one binary reference per annotator** from that worker's ratings using fixed 60-frame units and capacity `fix(0.15*N)`, then averages 20 machine–reference F1 values. Whole-shot selection can underfill the 15% capacity. No native split; later work often uses five random 80/20 partitions. | [Official repository](https://github.com/yalesong/tvsum), [CVPR 2015 paper](https://people.csail.mit.edu/yalesong/publications/SongVSJ2015CVPR.pdf), [official evaluator](https://github.com/yalesong/tvsum/blob/master/matlab/script_evaluate_result.m), [author download directory](https://people.csail.mit.edu/yalesong/tvsum/), [direct archive](https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz). |
| **OVP / VSUMM Open Video subset — legacy auxiliary** | **50** MPEG-1 videos at 30 fps and 352×240, with color and sound. The primary paper describes **1–4 min/video** and the author README says approximately **75 min total**. **Inherited archive audit, not repeated:** **4,981.877 s = 83:01.877**, range **46.923–209.162 s**. | **5 static keyframe summaries/video**. Fifty users each summarized five videos, giving **250 user summaries**. There are no temporal intervals, dense importance scores, or native percentage budget. | Original VSUMM uses content-based user-summary matching (CUS), color histograms, Manhattan distance, threshold 0.5, and CUSA/CUSE. seqDPP later reports content-matched keyframe precision/recall/F1 averaged across five users. Zhang et al. use all 50 videos only as auxiliary training data after deriving an oracle score sequence; there is no canonical split. | [Author repository](https://github.com/sandraavila/vsumm), [raw `database.zip`](https://www.dropbox.com/s/g0e64b4qfnuual1/database.zip?dl=1) (inherited, unverified MD5 `62e6b199b7c1dcfa6beed4dcbb83e46e`), [user summaries](https://www.dropbox.com/s/ilt1jpclzs2o18v/UserSummary.zip?dl=1), [Pattern Recognition Letters paper](https://ic.unicamp.br/~sandra/pdf/papers/avila_PRL11.pdf). |
| **YouTube / VSUMM web-video corpus — legacy auxiliary** | Original release: **50** cartoons, news, sports, commercials, TV shows, and home videos; the paper describes **1–10 min/video**. **Inherited archive audit, not repeated:** `newDatabase.zip` was measured at **8,705.383 s = 145:05.383** for all 50. Gong and Zhang instead use a **39-video non-cartoon subset**, totaling **7,834.253 s = 130:34.253** in that archive. The inherited audit found some files under one minute, so archive calculations and published ranges are reported separately. | **5 static keyframe summaries/video = 250 summaries**; no dense importance scores or native percentage budget. | Original CUS/CUSA/CUSE evaluation. Gong uses the 39-video subset with content-matched P/R/F and 100 random 80/20 trials; Zhang uses those 39 as auxiliary train/validation data. SUM-GAN describes the full 50-video corpus. A result must say which variant it used. | [Author repository](https://github.com/sandraavila/vsumm), [raw `newDatabase.zip`](https://www.dropbox.com/s/wxpj91cm9m3ikn0/newDatabase.zip?dl=1) (inherited, unverified MD5 `734962a1319dad0c93f9d5c38c50878d`), [user summaries](https://www.dropbox.com/s/7rtbyeo64hk8ot7/newUserSummary.zip?dl=1), [VSUMM paper](https://ic.unicamp.br/~sandra/pdf/papers/avila_PRL11.pdf), [seqDPP paper](https://www.cs.utexas.edu/~grauman/papers/nips14_seqdpp.pdf), [Zhang supplement](https://www.cs.utexas.edu/~grauman/papers/zhang-eccv2016-lstm-summ-supp.pdf). |
| **VideoXum** | ActivityNet-Captions reannotation: **14,001 videos**, **200 activity classes**; **10–755 s**, paper-reported mean **124.2 s**, median **121.6 s**, and 99.9% under 300 s. **Inherited JSON calculation, not repeated:** mean **124.285 s** and total **483.364 h**. | **10 independently annotated visual summaries/video**, binary masks over uniformly sampled **1-fps** positions; **140,010 visual/text pairs**. The event-caption list is shared across those visual annotations rather than ten independently written text summaries. Mean selected ratio **13.6%**, median **13.7%**, maximum **20%**. | Official split: **8,000 train / 2,001 validation / 4,000 test**. V2V: **F1-Avg and F1-Max across the ten visual references**, Kendall $`\tau`$, and Spearman $`\rho`$; V2T: BLEU-4, METEOR, ROUGE-L, CIDEr; cross-modal: VT-CLIPScore. Official V2V evaluation selects top-scoring frames directly—**no KTS or knapsack**. | [TMM paper](https://arxiv.org/abs/2303.12060), [project](https://videoxum.github.io/), [official code](https://github.com/jylins/videoxum), [official Hugging Face dataset](https://huggingface.co/datasets/jylins/videoxum). Raw video bytes are not bundled; follow the [ActivityNet/ActivityNet-Captions media-retrieval instructions](https://cs.stanford.edu/people/ranjaykrishna/densevid/). |
| **CoSum** | Query-specific co-summarization collection: **51 video instances**, **10 topical groups**, exact total **147 m 40 s**, **246,062 frames**, **2,762 shots**. | **3 judges** saw the query and selected **10–50% of shots/video**. Ground truth pools shots chosen by at least **2 of 3**. This is collection-aware query annotation, not dense generic importance. | Query-specific summarization uses **mAP@5 and mAP@15 shots**. Concept visualization also used a 20-person good/neutral/bad study. No canonical train/test split. | [Official repository](https://github.com/l2ior/cosum), [CVPR 2015 paper](https://people.csail.mit.edu/yalesong/publications/ChuSJ2015CVPR.pdf). The manifest supplies URLs, shot indices, and annotations—not durable raw video bytes; the inherited manifest audit reports repeated YouTube ID `jaXcwfsfAhk` for two “kids” instances and at least one non-YouTube host (not rechecked here), so do not assume 51 distinct, currently retrievable URLs without a local checksum/availability audit. |
| **UT Egocentric / UTE** | **4 public videos**, one per subject; authors captured 10 but withheld six for privacy. Each public recording is approximately **3–5 h**; [Plummer et al., CVPR 2017](https://slazebni.cs.illinois.edu/publications/cvpr17_summarization.pdf) report **over 17 h total**. Looxcie camera, **15 fps**, **320×480**. | **Inherited archive audit, not repeated:** sparse **320×480 binary important-region masks for 1,662 positive frames**, plus **1,057 negative frames**. The original release supplies important-region annotations, not dense temporal summary labels or multiple reference skims; the inherited mask counts require archive verification. The project explicitly says summary evaluation needs a new human-subject study. | Important-region prediction uses leave-one-video-out; IoU $`>0.5`$ is a true positive. Original summarization work uses human studies, so SumMe/TVSum F1 is not directly applicable. | [Official page](https://vision.cs.utexas.edu/projects/egocentric_data/UT_Egocentric_Dataset.html), [download registration](https://vision.cs.utexas.edu/projects/egocentric/download_register.html), [important-region annotations](https://vision.cs.utexas.edu/projects/egocentric_data/egocentric_GT.zip). The blurred-face video package is 1.4 GB. |
| **UTE subset of VideoSET annotations** | The full VideoSET release covers **11 videos / about 40 h**: four UTE daily-life videos, three Disneyworld egocentric videos, and four TV episodes. This row describes only the derivative annotation layer for the four UTE videos, not a separate raw-video corpus. | One factual sentence per non-overlapping **5-second segment**, plus **3 human reference text summaries/video**. Workers typically wrote and ranked **40–60 summary sentences/video**. | Selected segments are mapped to text and scored with **ROUGE-SU F-measure**, taking the maximum across references. Original UTE-subset evaluation constructs a 2-minute skim: 24 five-second segments. Later work uses four-fold leave-one-video-out. | [VideoSET paper](https://arxiv.org/pdf/1406.5824), [official project](https://ai.stanford.edu/~syyeung/videoset.html), [direct annotation ZIP](https://ai.stanford.edu/~syyeung/resources/videoset_data.zip), [Plummer et al., CVPR 2017](https://slazebni.cs.illinois.edu/publications/cvpr17_summarization.pdf). Raw UTE video remains in the UTE download. |
| **EgoSum+gaze** | Distinct from UTE: **21 videos** from **5 subjects**, each **15 min–1.5 h**, **over 15 h total**, captured with SMI/Pupil eye trackers. | Calibrated gaze plus summary annotations. Wearers select **5–15 semantic event blocks**; any subshot inside a block is an acceptable representative. Gaze segmentation yields roughly **6,000–9,000 fixation segments/hour**, grouped into about **800 subshots/hour**. | Event-aware precision/recall/F1: a proposed subshot is correct when it lies in a reference event block. No reusable canonical train/test split is specified. | [Official project](https://pages.cs.wisc.edu/~jiaxu/projects/ego-video-sum/), [CVPR 2015 paper](https://pages.cs.wisc.edu/~jiaxu/projects/ego-video-sum/ego-video-sum-cvpr2015.pdf). No public raw archive is currently linked. |
| **ActivityNet-QA — auxiliary** | **5,800** ActivityNet videos, mean duration **180 s**; split **3,200 train / 1,800 validation / 800 test**. The primary paper does not state an exact total or range. | Exactly **10 human QA pairs/video = 58,000**, spanning motion, spatial, temporal, and free-form questions. There are no reference skims or importance labels. | VideoQA accuracy and WUPS@0.0/WUPS@0.9; **not** a summarization benchmark. | [Official AAAI paper and DOI](https://ojs.aaai.org/index.php/AAAI/article/view/4946), [official QA repository](https://github.com/MILVLG/activitynet-qa), [ActivityNet parent site](https://activity-net.org/), and stable [ActivityNet crawler/media-retrieval instructions](https://github.com/activitynet/ActivityNet/tree/master/Crawler). The QA repository hosts annotations/evaluation, not video. |

### 2.1 Why ActivityNet-QA is not in the core leaderboard

ActivityNet-QA has no keyshots, importance curves, summary masks, user skims, or summary-duration target. It can support video-language pretraining or semantic reasoning diagnostics, but reporting summarization F1 on it would be a category error.

### 2.2 Query-focused and long-form resources

These additions support different learning paths. A dataset being useful for video understanding does not supply summary ground truth automatically.

| Resource | Corpus and annotation | Evaluation / split | Primary source and access |
|---|---|---|---|
| **QFVS — UTE annotation layer** | **4** UTE recordings, **3–5 h/video**; **48 concepts**, **46 queries/video**, **3 summaries/query-video pair**. Three workers tag each **5-second shot**; their tag union defines its semantic vector. | Concept-IoU matching between shot sets, followed by precision/recall/F1. Four rounds use **2 train / 1 validation / 1 test video**. Unconstrained references and later **10/20-shot** budgeted references are distinct. | [Sharghi, Laurel and Gong, CVPR 2017](https://arxiv.org/pdf/1707.04960), Sections 3 and 5. The original paper-linked project endpoint failed to fetch; annotation download is **not independently verified**. Raw video comes from [UTE](https://vision.cs.utexas.edu/projects/egocentric_data/UT_Egocentric_Dataset.html). |
| **SummScreen — adjacent, text-only** | Final ACL paper: **4,348 FD + 22,503 TMS = 26,851 episode records**; transcripts with community-written recaps. These are text records, not distributed videos. | FD train/dev/test **3,673/338/337**; TMS **18,915/1,795/1,793**. ROUGE, BLEU and character/relation metrics assess textual summaries. | [Chen et al., ACL 2022](https://aclanthology.org/2022.acl-long.589.pdf), Tables 2–3; [official repository](https://github.com/mingdachen/SummScreen) links both tokenized and untokenized data. The latter has extra filtered instances: recover final splits by filename matching. |
| **SummScreen3D — long video-to-text** | **4,575 episodes** from **5 shows**, approximately **40 min/video**; transcripts, audio/video and **1.53 recaps/episode** on average. Extends SummScreen-TMS. | **296 validation / 296 test**; remaining **3,983 training episodes** (calculated), expanded into **5,199 episode-summary training pairs**. ROUGE and entity/noun QA metrics assess episode recaps. | [Papalampidi and Lapata, EACL Findings 2023](https://aclanthology.org/2023.findings-eacl.96.pdf), Section 3 and Tables 2, 10; [official repository](https://github.com/ppapalampidi/long_video_summarization) has text, captions, splits and video downloader. Raw video retrieval was not tested. |
| **MoSu — multimodal, behavioral targets** | **52,678** YouTube8M-derived videos, **3,983.7 h**, **120–501 s/video**; visual, transcript and audio features. Targets derive from Most Replayed statistics. The selection threshold is **over 50,000 views**, not an independently verified count of summary annotators. | Topic-stratified **42,152 train / 5,263 validation / 5,263 test**. Kendall $`\tau`$, Spearman $`\rho`$, and segment mAP; replay targets measure engagement and require a separate interpretation from human-authored summaries. | [TripleSumm paper](https://arxiv.org/html/2603.01169v1), Section 4 and Appendix B; [official dataset card](https://huggingface.co/datasets/hminjeong/TripleSumm-MoSu). Metadata, targets, splits and three feature HDF5 files are advertised; raw videos remain source URLs. No binary download was tested. |
| **TripleSumm long-video test set** | **50** videos; mean **70.4 min**, range **2,413–7,207 s**; films, tutorials, full sports and talk shows with behavioral targets. | Evaluation-only direct transfer from MoSu; Kendall $`\tau`$ and Spearman $`\rho`$. | [Appendix B.6](https://arxiv.org/html/2603.01169v1): dedicated public artifact access is **not independently verified**. |
| **Ego4D — adjacent** | Version-dependent egocentric perception corpus; landing page reports **3,670 h**. Narrations and multiple task-specific annotations do not form one native generic-summary layer. | Splits and metrics depend on the named benchmark and release. A summarization derivative needs its own reference annotations and protocol. | [Official project](https://ego4d-data.org/) and [access documentation](https://ego4d-data.org/docs/start-here/). License approval and credentials precede dataset access; the CLI can retrieve selected subsets and feature packages. |
| **YouTube Highlights — adjacent** | Six domains; approximately **2-second clips** with matched/unmatched/borderline edited-video labels and separate MTurk votes. `vlist.json` and manually selected `vlist_sel.json` are different collections. | Highlight ranking AP/mAP; released train/test groups also distinguish tight and loose labels. Exact current counts were not reconstructed in this audit. | [Sun, Farhadi and Seitz's official release](https://github.com/aliensunmin/DomainSpecificHighlight). Video URLs, clip indices and labels are available; legacy media downloader was not run. This is **not** the VSUMM YouTube corpus. |
| **TVR / TVC — adjacent** | TVR reports **21.8K clips**, **109K temporal queries**, **6 shows**. TVC extends these moments with captions. | Moment retrieval / localized captioning, not episode recap generation or generic skim selection. | [Lei et al., ECCV 2020](https://arxiv.org/abs/2001.09099) and [official TVR project](https://tvr.cs.unc.edu/). Do not call a TVR-derived resource a summarization benchmark unless the specific summary annotation extension is identified. |

QFVS, VideoSET-UTE and native UTE share the same four underlying videos. SummScreen3D reuses SummScreen-TMS episodes. Keep these relationships in a leakage audit; counting annotation layers as independent corpora inflates apparent data diversity.

### 2.3 License and access are separate checks

| Release | Verified statement | What remains separate |
|---|---|---|
| [TVSum](https://github.com/yalesong/tvsum) | Dataset authors identify the video source license as **CC-BY 3.0**. | Later code/feature packages require their own license review and provenance. |
| [VideoXum dataset card](https://huggingface.co/datasets/jylins/videoxum) | Card declares **Apache-2.0**. | Underlying ActivityNet/YouTube media is not relicensed by that card. |
| [MoSu dataset card](https://huggingface.co/datasets/hminjeong/TripleSumm-MoSu) | Card declares **CC-BY-4.0**; code has a separate MIT license. | Source-video rights and the separate long-video test artifact require their own check. |
| [Ego4D](https://ego4d-data.org/docs/start-here/) | Data and annotations require the Ego4D license agreement and approved access. | A public code repository does not remove the data-access requirement. |
| Other releases above | **Not independently verified** where no explicit artifact license was located. | A citation request or a downloadable archive is not a general reuse license. Repository MIT licensing covers this handbook, not third-party artifacts. |

## 3. Annotation geometry at a glance

| Dataset/resource | Native annotation unit / common model unit | Reference form | References/video | Budget or output size | Audio during annotation |
|---|---|---|---:|---:|---|
| SumMe | User-selected temporal intervals / later sampled-frame scores decoded to shots | Temporal intervals | 15–18 | Human summaries 5–15%; machine summaries usually ≤15% | Muted |
| TVSum | Uniform 2-second annotated shots / later frame scores decoded with KTS | Per-shot 1–5 importance ratings | 20 | 15% after decoding | Muted |
| OVP | Selected keyframes | Static keyframe sets | 5 | No fixed percentage budget | Not stated |
| YouTube / VSUMM | Selected keyframes | Static keyframe sets | 5 | No fixed percentage budget | Not stated |
| VideoXum | Uniformly sampled 1-fps positions | Binary selected positions/spans | 10 | Mean 13.6%; curation cap 20% | Not stated |
| CoSum | Presegmented shots | Query-relevant selected shots | 3 judges, pooled by majority | Annotation range 10–50%; evaluation at top 5 and 15 shots | Not stated |
| UTE original | Spatiotemporal important regions | Important-region annotations; no reusable reference skims | N/A | N/A | Not stated |
| UTE subset of VideoSET | Non-overlapping 5-second segments | One sentence/segment plus human text summaries | 3 text summaries | Original UTE-subset evaluation: 24 segments = 2 minutes | Not stated |
| EgoSum+gaze | Gaze-derived subshots grouped into semantic event blocks | Wearer-selected event blocks | One wearer-derived gold standard/video | 5–15 event blocks; no universal percentage budget | Not stated |
| ActivityNet-QA | Open-ended answers, not summaries | Question–answer pairs | 10 QA pairs | None | Not stated in the primary paper |

## 4. CoSum group statistics

<details>
<summary>Expand exact query-group counts</summary>

| Query group | Video instances | Combined duration | Frames | Shots |
|---|---:|---:|---:|---:|
| Base jumping | 5 | 10:54 | 17,960 | 241 |
| Bike polo | 5 | 14:08 | 22,490 | 341 |
| Eiffel Tower | 7 | 25:47 | 43,729 | 381 |
| Excavators river crossing | 3 | 10:41 | 16,019 | 112 |
| Kids playing in leaves | 6 | 15:40 | 27,972 | 238 |
| MLB | 6 | 12:11 | 21,271 | 201 |
| NFL | 3 | 13:28 | 23,179 | 405 |
| Notre Dame Cathedral | 5 | 11:26 | 20,110 | 196 |
| Statue of Liberty | 5 | 10:44 | 18,542 | 164 |
| Surfing | 6 | 22:40 | 34,790 | 483 |
| **Total** | **51** | **147:40** | **246,062** | **2,762** |

</details>

## 5. Raw-video and feature-artifact audit

Legend:

- **Official** — dataset or benchmark authors.
- **Paper-author derivative** — authors of a later summarization paper.
- **Not found** — no ready-made tensor link in audited first-party releases; this does not prove no private or community extraction exists.

| Dataset | Raw video status | GoogLeNet pool5 | ResNet-101 | I3D | CLIP-ViT | VideoMAE | Other verified features |
|---|---|---|---|---|---|---|---|
| **SumMe** | **Official:** [SumMe.zip](https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip) | **Paper-author derivative:** [Zhang et al., ECCV 2016](https://www.cs.utexas.edu/~grauman/papers/zhang-eccv2016-lstm-summ.pdf) specify the 1,024-D GoogLeNet `pool5` representation, while the [repository](https://github.com/kezhang-cs/Video-Summarization-with-LSTM) documents 2-fps HDF5 extraction; a separate paper-author repackaging by Kanafani et al. is archived at [Zenodo 4884870](https://zenodo.org/records/4884870) | **Not found** | **Paper-author derivative:** RGB/flow `.npy` from the [MSVA authors' repository](https://github.com/TIBHannover/MSVA), archived at [Zenodo 4682137](https://zenodo.org/records/4682137) | **Not found as ready-made official tensors** | **Not found** | Original release has no deep tensors |
| **TVSum** | **Official:** [author directory](https://people.csail.mit.edu/yalesong/tvsum/) includes files and source URLs | Same ECCV 2016 / [Zenodo 4884870](https://zenodo.org/records/4884870) derivative | **Not found** | Same MSVA / [Zenodo 4682137](https://zenodo.org/records/4682137) derivative | **Not found as ready-made official tensors** | **Not found** | Original release has annotations/evaluation but no deep tensors |
| **OVP** | **Author release:** [`database.zip`](https://www.dropbox.com/s/g0e64b4qfnuual1/database.zip?dl=1) | **Paper-author derivative advertised; inherited audit reported download unavailable:** Zhang's repository describes 2-fps, 1,024-D pool5 HDF5 | **Not found** | **Not found** | **Not found** | **Not found** | **Paper-author derivative:** seqDPP released 1-fps, L2-normalized 8,192-D SIFT Fisher vectors plus `fishers_PCA90`, saliency, context, and histograms; see the direct bundle below |
| **YouTube / VSUMM** | **Author release:** [`newDatabase.zip`](https://www.dropbox.com/s/wxpj91cm9m3ikn0/newDatabase.zip?dl=1) | **Paper-author derivative advertised; inherited audit reported download unavailable:** Zhang's repository describes 2-fps, 1,024-D pool5 HDF5 for the 39-video variant | **Not found** | **Not found** | **Not found** | **Not found** | Same seqDPP feature bundle for the 39-video variant; original VSUMM release contains no deep tensors |
| **VideoXum** | **Not bundled**; obtain ActivityNet media separately | **Not found** | **Not found** | **Not found** | **Official:** fine-tuned **VT-CLIP ViT-B/16** [feature archive](https://huggingface.co/datasets/jylins/videoxum/resolve/main/vt_clip_feat.zip?download=true)—not generic frozen CLIP | **Not found** | **Official:** [BLIP features](https://huggingface.co/datasets/jylins/videoxum/resolve/main/blip_feat.zip?download=true) and [VTSUM-BLIP checkpoints](https://huggingface.co/jylins/vtsum_blip/tree/main) |
| **MoSu** | Source YouTube IDs; raw bytes not bundled | **Not found** | **Not found** | **Not found** | **Official:** CLIP ViT-L/14 HDF5, 768-D at 1-second intervals | **Not found** | **Official:** RoBERTa-base and AST, each 768-D; [card and files](https://huggingface.co/datasets/hminjeong/TripleSumm-MoSu/tree/main) advertise approximately 40 GB per modality, 136 GB total; integrity not checked |
| **CoSum** | Official repository supplies YouTube URLs, not bytes | **Not found** | **Not found** | **Not found** | **Not found** | **Not found** | Paper computes 254-D CENTRIST + 3,840-D Dense-SIFT + 108-D HSV moments, then L2-normalizes/PCA-reduces to 400-D; official repo does not advertise tensors |
| **UTE** | **Official:** 1.4-GB package via registration | **Not found** | **Not found** | **Not found** | **Not found** | **Not found** | Plummer et al. used ResNet-152—not ResNet-101—but did not publish a tensor archive in the paper source |
| **EgoSum+gaze** | No raw archive linked by official project | **Not found** | **Not found** | **Not found** | **Not found** | **Not found** | Paper uses R-CNN descriptors and gaze-derived segmentation; no first-party tensor archive linked |
| **ActivityNet-QA** | QA repo does not host video | **Not found** | **Not found** | **Not found** | **Not found** | **Not found** | Paper extracted VGG-16 fc7 and C3D fc7, but official QA repo does not publish a tensor bundle |

### 5.1 Feature-bundle files and inherited integrity metadata

Landing pages are useful for citation and metadata, but a reproducible registry also needs the immutable file endpoint and checksum. The sizes and MD5 values below are inherited from the earlier handbook audit of the corresponding Zenodo API records; neither archive was downloaded or independently hashed in this run.

| Artifact | Applies to | Exact file | Size | Integrity |
|---|---|---|---:|---|
| Kanafani et al. benchmark package | SumMe, TVSum, and additional legacy datasets; includes the commonly reused GoogLeNet HDF5 data | [`datasets.tar` direct download](https://zenodo.org/api/records/4884870/files/datasets.tar/content) · [record 4884870](https://zenodo.org/records/4884870) | 3,231,123,456 bytes | MD5 `01493214a8b775540a4f6a9d9dcd8d95` |
| MSVA package | SumMe and TVSum GoogLeNet plus I3D RGB/flow arrays | [`msva_video_summarization.tar` direct download](https://zenodo.org/api/records/4682137/files/msva_video_summarization.tar/content) · [record 4682137](https://zenodo.org/records/4682137) | 4,366,576,640 bytes | MD5 `7ad3c33ca478336292fed723bd0d7f71` |
| seqDPP author package | OVP and the 39-video YouTube variant; oracle/ground sets, Fisher features, saliency, context, and histograms | [`video_summarization.zip` direct download](https://www.dropbox.com/s/bpy41o2zglk4ka4/video_summarization.zip?dl=1) · [paper-linked repository](https://github.com/pujols/Video-summarization) | Not published | No checksum published; record one after download |

The accurate shorthand is:

> The original SumMe and TVSum distributions do not include deep features. Widely used GoogLeNet pool5 HDF5 files and I3D RGB/flow arrays are later paper-author packages, not dataset-owner artifacts.

The ECCV 2016 repository states that features were extracted from videos downsampled to **2 fps** and uses names such as `Data_$Dataset$_google_p5.h5`. The MSVA authors' repository documents:

```text
datasets/object_features/eccv16_dataset_summe_google_pool5.h5
datasets/object_features/eccv16_dataset_tvsum_google_pool5.h5
datasets/kinetic_features/summe/RGB/features/*.npy
datasets/kinetic_features/summe/FLOW/features/*.npy
datasets/kinetic_features/tvsum/RGB/features/*.npy
datasets/kinetic_features/tvsum/FLOW/features/*.npy
```

No audited **dataset-author release or cited paper-author package** supplied ready-made **ResNet-101** or **VideoMAE** tensors for these benchmarks.

## 6. Artifact-manifest requirements

A feature file is reproducible only if its manifest records:

| Field | Why it matters |
|---|---|
| Raw video identifier and checksum | Protects against YouTube replacement, recoding, and filename collisions |
| Decoder/version and time base | Variable-frame-rate decoding changes frame indices |
| Exact sampled indices and timestamps | “2 fps” alone does not define rounding or the first sample |
| Backbone, checkpoint URL/hash, layer | Architecture names do not identify weights or endpoint |
| Resize, crop, color, normalization | Preprocessing materially changes embeddings |
| Tensor shape, dtype, pooling, L2/PCA | Prevents silent dimensional and metric mismatches |
| Clip window/stride/center | Required for I3D and VideoMAE temporal alignment |
| If shot/KTS decoding is used: segmentation input, implementation/version, and boundary file/hash | Segmentation is a major source of variance; this is not applicable to native VideoXum evaluation |
| Split IDs and visible training videos | Needed to detect leakage and transfer/augmentation |

HDF5 packages may colocate `features`, `gtscore`, `gtsummary`, `user_summary`, `change_points`, and split metadata. Unsupervised training code must prove that human-derived fields are not used for optimization, early stopping, checkpoint choice, or hyperparameter tuning.

## 7. Protocol cautions by dataset

- **SumMe and TVSum have no original canonical train/test split.** [Zhang et al., ECCV 2016](https://www.cs.utexas.edu/~grauman/papers/zhang-eccv2016-lstm-summ.pdf) document the random **80/20 canonical setting**; packaging this as **five pre-generated JSON splits** is a later implementation convention, explicitly documented for example by [SUM-GAN-AAE](https://github.com/e-apostolidis/SUM-GAN-AAE#training).
- **VideoXum is frame-based.** Its released evaluator ranks 1-fps positions directly. Adding KTS and knapsack silently changes the benchmark.
- VideoXum's released [`get_top15_frames()` threshold](https://github.com/jylins/videoxum/blob/11bd4fe3fb51b7dc804dd1519a4914f67bff60df/utils.py#L275) uses the element at index $`\lfloor0.15T\rfloor`$ and `>=`; unique scores therefore select $`\lfloor0.15T\rfloor+1`$ positions, while ties can select more.
- VideoXum's released [correlation evaluation](https://github.com/jylins/videoxum/blob/11bd4fe3fb51b7dc804dd1519a4914f67bff60df/eval_v2vt_sum.py#L61) compares predictions with **mean-annotator importance**, rather than averaging separate per-user correlations. Preserve the released implementation when reproducing its results; do not substitute the per-user ranking diagnostic in the evaluation chapter without declaring the change.
- **OVP and YouTube/VSUMM are static-keyframe datasets.** Their content-matched CUS F-score is not temporal-overlap F1, and Zhang's oracle/score conversion is a derived training transformation rather than native annotation. For YouTube, name the original 50-video corpus or the Gong/Zhang 39-video subset (archive IDs 71–81 and 83–110).
- **UTE, the UTE subset of VideoSET, and EgoSum+gaze are distinct resources.** Their annotations and metrics are not interchangeable; the full VideoSET release also contains seven non-UTE videos.
- **CoSum is query-specific and reports mAP**, not SumMe/TVSum temporal-overlap F1.
- For any URL-based collection, record whether the cited artifact hosts bytes, URLs only, annotations only, or derived tensors, plus the last verification date.
