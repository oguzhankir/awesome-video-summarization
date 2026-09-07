# Field overview and historical timeline

[Home](../README.md) · [Learning path](16-learning-path.md) · [Taxonomy](05-taxonomy.md)

A summary should retain what matters while reducing viewing or reading time. What matters depends on the intended output and user: a diverse storyboard, an instructional recap, and a query-specific clip list need different targets. Begin by defining that target, then choose a dataset and protocol before choosing a model.

## Milestones to read in order

| Period | Representative primary reading | What changed |
|---|---|---|
| Classical selection | [VSUMM (2011)](https://ic.unicamp.br/~sandra/pdf/papers/avila_PRL11.pdf) | Color/cluster-based keyframes and user-summary matching |
| External visual priors | [Web-image priors (CVPR 2013)](https://openaccess.thecvf.com/content_cvpr_2013/html/Khosla_Large-Scale_Video_Summarization_2013_CVPR_paper.html) | Selection incorporates external visual evidence |
| Structured subset learning | [seqDPP (NeurIPS 2014)](https://www.cs.utexas.edu/~grauman/papers/nips14_seqdpp.pdf) | Supervised diversity with temporal structure |
| Shared benchmarks | [SumMe (ECCV 2014)](https://doi.org/10.1007/978-3-319-10584-0_33), [TVSum (CVPR 2015)](https://people.csail.mit.edu/yalesong/publications/SongVSJ2015CVPR.pdf) | Multiple human references and repeatable evaluation |
| Learned summary objectives | [Submodular mixtures (CVPR 2015)](https://openaccess.thecvf.com/content_cvpr_2015/html/Gygli_Video_Summarization_by_2015_CVPR_paper.html) | Supervision learns how to combine summary properties |
| Deep sequence selection | [vsLSTM/dppLSTM (ECCV 2016)](https://arxiv.org/abs/1605.08110) | Temporal score prediction and diversity learning |
| Label-free selector objectives | [SUM-GAN (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.html), [DR-DSN (AAAI 2018)](https://arxiv.org/abs/1801.00054) | Reconstruction or explicit subset rewards |
| Faster temporal architectures | [VASNet (ACCV 2018 workshop)](https://arxiv.org/abs/1812.01969), [PGL-SUM (ISM 2021)](https://github.com/e-apostolidis/PGL-SUM) | Attention and global/local context |
| Evaluation scrutiny | [Rethinking evaluation (CVPR 2019)](https://arxiv.org/abs/1903.11328) | Post-processing can dominate F1; rank metrics expose score quality |
| Multimodal supervision | [VideoXum (2023 preprint / TMM 2024)](https://arxiv.org/abs/2303.12060), [A2Summ (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/He_Align_and_Attend_Multimodal_Summarization_With_Dual_Contrastive_Losses_CVPR_2023_paper.html) | Visual/textual targets and audio alignment |
| Generated and prompted summaries | [SummDiff (ICCV 2025)](https://arxiv.org/abs/2510.08458), [TripleSumm (ICLR 2026)](https://arxiv.org/abs/2603.01169) | Annotation distributions and adaptive three-modality fusion |

This timeline is selective, not a claim that every change began with the listed paper. Dates distinguish preprint release from proceedings/journal publication.

## Choosing a starting point

For a first implementation, study the [budget decoder example](../examples/summary_baselines.py), then one supervised attention model and one reward-based model on the same feature/split package. Read the [evaluation chapter](02-evaluation.md) before comparing their numbers. For narrative text or long-video work, start with the [modern systems chapter](10-foundation-models.md); the SumMe/TVSum skim pipeline cannot be assumed to apply.

The current research direction combines stronger encoders, richer audio/text information, and explicit user intent. The remaining challenge is showing that improvement survives controlled features, protocols, human preference variation, and realistic inference cost. These are questions to test, not reasons to crown a single global winner.
