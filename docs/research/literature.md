# 文献、上游项目与可迁移思想

调研日期：2026-10-07。范围是本届赛题的音频/视频真假判别、未知生成器泛化、EER 和离线交付。以下是对原论文方法、实验边界与上游实现的归纳；没有复现论文指标，也没有测出本届 EER。论文名、年份和来源是证据，候选顺序是针对本项目资源的判断。

## 音频

| 文献 / 项目 | 阅读与核验范围 | 可迁移内容与采用边界 |
| --- | --- | --- |
| [AASIST](https://arxiv.org/abs/2110.01200)，2021 预印本 / ICASSP 2022；[作者代码](https://github.com/clovaai/aasist) | 方法与实验、5 页 PDF；检查模型配置、训练/评测入口及 MIT 许可 | 以谱域和时域的图注意力整合伪迹；AASIST-L 约 85K 参数，保留为语音对照。原配方依赖旧环境，不能把 ASVspoof 2019 成绩当作本届泛化结果。 |
| [ASVspoof 2021](https://www.asvspoof.org/index2021.html)；[官方工具](https://github.com/asvspoof-challenge/2021) | DF/LA 数据说明、协议与 `eval-package/eval_metrics.py` 的 DET/EER 实现 | 学习样本协议、评分方向及跨编解码条件评测。其离散 EER 算法是交叉核对参考，不能取代本届官方实现；逐子目录核对工具许可。 |
| [ASVspoof 5 资源设计](https://arxiv.org/abs/2502.08857)，2025；[评测计划 v0.6](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf) | 数据设计、说话人互斥分区、未知及对抗攻击协议 | 使用更接近现实采集条件和未知攻击的留出设计。主任务与指标组合不同，借鉴协议，不把 min-tDCF/minDCF 直接替换本赛 EER。 |
| [XLS-R + SLS 方法论文](https://openreview.net/attachment?id=acJMIXJg2u&name=pdf)，ACM MM 2024；[复现仓库](https://github.com/Yash-Sukhdeve/XLS-R-SLS-Deepfake-Detection)；[XLS-R 上游](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr) | 层级表征加权、RawBoost 与训练策略 | 第二条语音候选采用冻结 SSL 特征与轻量头，再比较少量层解冻。复现仓库不是已确认可直接提交的作者基线；fairseq 旧依赖与 Python 3.12 的迁移成本须先测量。 |
| [MLAAD](https://arxiv.org/abs/2401.09512)，2024；[维护方数据页](https://deepfake-demo.aisec.fraunhofer.de/mlaad) | 多语种/TTS 来源覆盖及数据许可 | 用于语言和生成器多样性设计。伪造数据存在 CC BY-NC 条款，真实音频有独立来源许可；未确认参赛适用性前不纳入训练包。 |
| [In-the-Wild](https://arxiv.org/abs/2203.16263)，Interspeech 2022；[维护方页面](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) | 真实网络语音采集和跨域失效分析 | 提醒实验室语音反欺骗模型可能在网络传播条件下退化。适合作为事先冻结的外部留出候选；仓库/数据说明许可和原录音来源权利需分别核对。 |
| [AI-Synthesized Voice Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/34221)，AAAI 2025；[作者代码](https://github.com/Purdue-M2/AI-Synthesized-Voice-Generalization) | 域无关特征、无关内容解耦及泛化设计 | 可借鉴“控制内容/背景捷径”的实验思想；代码带非商业限制，当前不纳入主提交代码。 |
| [EBM](https://doi.org/10.21437/Interspeech.2025-422)，Interspeech 2025；[作者代码](https://github.com/kimho1wq/EBM) | 真实语音嵌入约束与一类建模方向 | 后续损失函数消融候选；先验证真实域差异是否使一类假设失效，不因较新就替换可复现基线。 |
| [Codecfake](https://arxiv.org/abs/2405.04880)，2024 | 问题定义与 neural-codec/Audio-LM 威胁描述 | 提醒只覆盖旧声码器不足以支持现代音频生成。数据、代码和授权核验尚不足，保留为继续调查项。 |

这些资料主要覆盖**语音**。它们不能证明对音乐、环境声、音效或混合音频有效。方案中的内容类型审计是必须经过的分支条件：若发现非语音，就建立单独评测桶及通用音频候选，不将语音子集 EER 冒充整个音频任务成绩。

## 视频与通用生成图像

| 文献 / 项目 | 阅读与核验范围 | 可迁移内容与采用边界 |
| --- | --- | --- |
| [DeepfakeBench](https://papers.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf)，NeurIPS 2023；[代码](https://github.com/SCLBD/DeepfakeBench) | 统一预处理、评测协议与仓库帧/视频指标 | 借鉴可比实验、视频级评测及记录方式。大量数据以人脸操纵为中心，CC BY-NC 代码不能默认视为本届可直接复制的实现。 |
| [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html)，CVPR 2023；[代码](https://github.com/Yuheng-Li/UniversalFakeDetect) | 未知扩散生成器失效分析、冻结 CLIP 线性探测及 JPEG/模糊实验 | 首个低成本空间基线可采用冻结视觉特征与轻量头。图像结果不是视频时序证据；仓库许可证仍需进一步确认，思想可借鉴而代码暂不直接引入。 |
| [Community Forensics](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html)，CVPR 2025；[训练/评测代码](https://github.com/s-jyeon/CommunityForensicsTrain) | §3–4 的生成器多样性、固定样本预算对比、JPEG/模糊实验；检查 224/384 checkpoint 加载入口与 MIT 许可 | 借鉴多生成器覆盖、对称增强和可缓存空间特征；旧项目也用过相关检测器。完整数据达 TB 量级，当前不下载；代码许可不自动覆盖所有数据和权重。 |
| [AI-Generated Video Detection via Spatio-Temporal Anomaly Learning](https://arxiv.org/abs/2403.16638)，PRCV 2024；[AIGVDet](https://github.com/multimediaFor/AIGVDet) | RGB/RAFT 光流两分支、分数级融合、视频平均及 H.264 实验 | 证明应单独检验时间信息，而非只检测人脸。光流的成本与较旧生成器覆盖限制使它成为后续消融；学术/非商业条款须确认。 |
| [Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale](https://proceedings.iclr.cc/paper_files/paper/2026/hash/20e45668fefa793bd9f2edf19be12c4b-Abstract-Conference.html)，ICLR 2026；[作者代码](https://github.com/mgiant/Qwen2.5ViT-AIGVDetection) | 固定缩放丢失伪迹的问题、可变空间尺度和时长、公开推理与 checkpoint 入口 | 优先把保比例、局部裁块与多尺度纳入轻量消融。Qwen 视觉主干、原尺度视频模型的当前资源可行性未测，许可证未明确，不作为首版依赖。 |

以上空间/视频研究以各自 AUC、Accuracy 或特定协议报告性能。是否能降低本赛 EER，必须在固定的未知生成器、来源与压缩留出集上重新比较。

## 数据来源候选登记

| 候选 | 计划用途 | 采用前必须补齐 |
| --- | --- | --- |
| ASVspoof 2019/2021/5 | 语音训练/开发与跨攻击验证 | 每套数据许可、官方 split、来源与说话人交叉关系；保持开发与最终留出分离 |
| MLAAD | 中文/多语言/TTS 泛化候选 | 版本、真实源、非商业许可及生成器互斥划分 |
| In-the-Wild | 外部网络语音留出候选 | 原媒体授权、人物/来源去重；冻结后不反复用于调参 |
| Community Forensics 小规模合法子集 | 通用视觉帧能力与生成器多样性 | 数据、权重各自授权，容量预算，真实源匹配与不重复抽样 |
| AIGVDet / Native Scale 相关视频数据 | 真正生成视频的时序验证候选 | 下载可用性、媒体使用条款、生成器与来源标识、预算；不把名称当作已取得数据 |
| 自建或团队合法持有数据 | 补齐非语音、现代生成器及传播域 | 来源同意、生成器版本、采集处理流程、是否与官方数据重叠 |

这些是候选登记，不是已通过授权审核的数据集名单。本轮仅预览了 `jungjee/asvspoof5` 的 split 元数据，没有下载该数据集。

## 对旧项目的取舍

参考 [旧系统总说明](../../archive/legacy-20261005/README.md)、[视频分带失效证据](../../archive/legacy-20261005/upgrade-research/docs/M4.1_视频级分带校准.md) 和 [旧评测器说明](../../archive/legacy-20261005/validation/deepfake_acceptance_v2/evaluator/README.md)：

- 保留的思想：模型常驻避免反复加载、确定性抽帧、逐样本日志、失败显式记账、数据清单与哈希、跨域验证。
- 新实现不继承：Windows 打包运行时、三模块 UI、身份/底图检索、图像风险分带阈值、旧自定义路径和不完整训练工程。
- 旧视频跨语料阈值失效是需要重建验证协议的直接证据，不是通过新增 UI 或固定阈值可以解决的问题。

## 研究结论与待验证假设

当前顺序是先完成可比音视频基线，再按实际结果增加容量和时间信息；使用已分配的RTX4090。具体优先级见[路线图](../roadmap.md)。待验证假设是：多生成器覆盖、对称压缩/重采样增强、稳健媒体级聚合、冻结预训练表征和轻量时间信息，能以较低资源改善未知域 EER。未通过消融前不称其为有效优化或本届最佳方案。
