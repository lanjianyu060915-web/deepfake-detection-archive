# 数据获取候选与许可闸门

核对日期：2026-10-07。范围是第七届中国人工智能大赛 AIGC 音视频伪造识别赛的
**本地开发数据**。本记录的初筛阶段只读取公开来源；实际获取进度以
[项目状态](../status.md) 为准。预训练资源另见 [固定资源记录](pretrained-resources/README.md)。

本轮实际使用项目的 [`huggingface-datasets`](../../.agents/skills/huggingface-datasets/SKILL.md)
skill：按其只读 Dataset Viewer/数据卡流程核实公开数据卡、split、字段、大小和许可。
初筛时普通沙箱 DNS 请求失败；随后在获授权的联网环境中取得了 Viewer 响应和固定
来源索引，具体证据随各项研究归档。实际下载仍须固定版本、清单、大小与哈希，不能
把数据卡信息当作已经获取的媒体。

## 赛方数据与接口状态

- 复核入口：[赛事总页](https://ai.xm.gov.cn/competition/competition-detail.html?id=2026c40dbb2347fba8b3c9a6294efa5b)、[赛题页](https://ai.xm.gov.cn/competition/project-detail.html?id=2026c40item0004&competeId=2026c40dbb2347fba8b3c9a6294efa5b)。
  公共页证实音频、视频各约一万条、均以 EER 评测以及安全屋离线执行；与
  [`requirements.md`](../competition/requirements.md) 一致。
- 当前公开页没有样例媒体、下载清单、安全屋镜像、输入目录/格式、输出列、分数方向、
  时限、外部数据或权重申报细则。不能把这一观察写成“主办方尚未发布”；它们可能只在
  报名后平台出现。
- 因此，官方测试集绝不用于本地选模；视频音轨也不混入独立音频赛道，直到赛方 I/O
  契约明确。

## 判定标准与预算

首批总媒体预算不超过 30 GiB，缓存不超过 20 GiB，项目累计不超过 100 GiB；本地为
RTX 4060 Laptop 8 GiB。候选必须同时满足：媒体本身可公开 GET、许可和上游来源足以
支持当前登记用途（本地研究与正式竞赛分别验收）、真伪标签可追溯，且已知同源媒体及
生成派生物不能跨划分。未知关系显式保留，不能冒充已证实的统计独立组。

每条登记均保存：`dataset_id`、版本/commit、直接 URL、下载日期和 SHA-256、原媒体 ID、
`label`、标签语义、`split`、`group_id`、真实来源、生成器/版本、媒体参数和许可。目录名
不是标签；文件能下载也不是许可。

## 可立即登记、待分片选择的音频主候选：EnvSDD

| 项目 | 已核实信息 | 首批用途与容量结论 |
| --- | --- | --- |
| 入口 | [EnvSDD/EnvSDD 数据卡](https://huggingface.co/datasets/EnvSDD/EnvSDD)，DOI [10.57967/hf/9156](https://doi.org/10.57967/hf/9156)，作者也链接 [开发集 Zenodo](https://doi.org/10.5281/zenodo.15220951) 与 [测试集 Zenodo](https://doi.org/10.5281/zenodo.15241138) | 公开，无登录门槛见数据卡；实际脚本先调用 skill 所列 `/is-valid` 和 `/splits` 固化 repo ID。数据卡示例中的 `AustinXiao/EnvSDD` 与页面实际 ID 不一致，不能照抄该示例。 |
| 许可 | 数据卡为 **CC BY 4.0**；同时明确要求遵守 UrbanSound8K、DCASE 2023、TAU UAS 2019、TUT SED 2016/2017、Clotho 等真实源的各自许可 | 可先进行来源许可二次登记；未把“数据卡 CC BY”扩大解释为覆盖所有原始音频。 |
| 内容 | 45.25 小时真实环境声、316.74 小时 AIGC 环境声；16 kHz、每条 4 秒；5 个 text-to-audio 与 2 个 audio-to-audio 生成器；单声道和复声场景 | 这是非语音的真伪二元候选，适合补齐语音反欺骗的盲区，也适合 log-mel/冻结音频表征基线。 |
| 标签与溯源 | 字段：`audio`、`label`（`real`/`fake`）、`attack_type`（`TTA`/`ATA`/`-`）、`generative_model`、`dataset`（真实源）、`filename`、`scene`、`event`、`caption` | `dataset`/`filename` 是分组和泄漏控制的首选键；同一原始音频及其多种生成版本必须落在同一 split。`generative_model` 用于未知生成器留出，不能从 `label` 反推。 |
| 公布 split | train：139,055 条、25,899,407,707 B（约 24.12 GiB）；validation：39,710 条、7,397,011,712 B（约 6.89 GiB）；test：39,768；remain：107,259 | train+validation 约 31.01 GiB，违反首批 30 GiB 上限，也超过 20 GiB 缓存。首批只可从 train 选择具固定 `group_id` 的小分片，且不能为凑容量跨原始来源。 |

**首批获取方案（未执行）**：先下载/读取元数据，以 `(dataset, filename)` 关联切片的生成
派生，再向原始录音合并；UrbanSound8K 按官方命名合同使用 `(dataset, fsID)` 划分。按
真实源、`scene/event`、标签、生成器分层；选定若干完整 group 至多约 10 GiB 原媒体，留
出若干完整真实源和至少一个生成器作外部开发留出。使用 Hub 的分片 URL/API，仅下载被
选行所属分片；下载前逐一核对分片累计字节数，避免 `load_dataset(...)` 或
`snapshot_download(...)` 无筛选拉取 24 GiB train。确认本地真实源许可后，才能把该集合
标为可训练集。

固定 `datasplit_dev.csv` 的实测审计见 [录音组审计](envsdd-group-audit.json)：
UrbanSound8K 的 validation 380 个原录音中有 369 个也出现在 train。
这是该 CSV 的 `usage` 交集，**尚未证明与当前 Hub 发布 split 一一对齐**；
不得直接沿用它做独立开发集。可用 [原始审计脚本](source/envsdd-group-audit.py) 对固定
SHA-256 的 CSV 重算；其元数据审计不涉及媒体下载或模型表现。

来源依据：[EnvSDD 数据卡的规模、生成类型和字段](https://huggingface.co/datasets/EnvSDD/EnvSDD)，
[其许可及上游许可提示](https://huggingface.co/datasets/EnvSDD/EnvSDD)。

## 小规模音频压力测试候选：FakeSound

| 项目 | 已核实信息 | 使用边界 |
| --- | --- | --- |
| 入口与许可 | [ZeyuXie/FakeSound](https://huggingface.co/datasets/ZeyuXie/FakeSound)，Apache-2.0 | 可公开取得；数据卡显示 1,899 条、约 987 MB。 |
| 内容 | 10 秒通用音频伪造片段；论文说明基于 AudioCaps 的原音频，以 AudioLDM/AudioLDM2 进行片段重建 | 覆盖环境声/通用音频，而非仅语音。 |
| 标签 | 卡片的 4 个类别是 `easy`、`hard`、`zero-shot` 等伪造难度/条件；上游 README 的 `label=0` 表示 deepfake | 此发布物不应误记为完整 binary real/fake 数据集；不能把难度类标签直接当比赛标签，也不能用不配对真实音频拼成训练集。 |

推荐把它作为首个 **fake-only 外部诊断集**：模型在 EnvSDD 上训练时，对 FakeSound 只计算
伪造分数分布、召回和失败样例，不用来选择最终阈值；需二元 EER 时，必须使用经许可并在
原始媒体层面配对的 bona fide 数据。

另有 [SynSFX-fake](https://huggingface.co/datasets/DGQW/SynSFX-fake)：非语音 fake-only，约
27.5k WAV、约 20 GB，包含 7 个 TTA 生成器与额外 ElevenLabs，CSV 含 `split`、`source`、
`rel_path`。但尚未取得同级真实集的清晰许可及字段证据，故不入首批；未来必须同
[SynSFX-real](https://huggingface.co/datasets/DGQW/SynSFX-real) 的来源/组对应关系一起审核，
不能把生成器 `source` 混作 binary `label`。

依据：[数据卡](https://huggingface.co/datasets/ZeyuXie/FakeSound)、
[作者的原始数据说明](https://github.com/FakeSoundData/FakeSound)、
[论文](https://arxiv.org/abs/2406.08052)。

## 语音真伪数据：可执行的开发与留出路径

语音数据用于补足 EnvSDD 不覆盖的 TTS/VC 条件，但其指标只能写为“语音子集 EER”，
不得替代全类型音频赛道 EER。

| 集合 | 内容、规模、获取 | 许可、分组与用途 |
| --- | --- | --- |
| [ASVspoof 2019 LA（官方 DataShare）](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45/full) | 直接档案 [`LA.zip`](https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download)，7.12 GB，MD5 `30c98f11d8b2bc21f2c257bfd78bb5c5`。训练为 25,380 条（2,580 bona fide + 22,800 spoof），开发为 24,844（2,548 + 22,296）；LA 覆盖 TTS/VC。 | 官方 `dc.rights` 为 **ODC-BY**，但许可证文本说明不自动覆盖单个内容；record 标明 VCTK 来源，故保留归属、协议、speaker/attack 与档案哈希。官方划分为 train 20、dev 10、eval 48 位说话人且 speaker-disjoint；开发集只按官方 protocol 使用，绝不从 2019 eval 选超参。 |
| [ASVspoof 2021 LA](https://zenodo.org/records/4837263) | 公开、无需登录的评测档案，7.8 GB；标签与元数据另由 [官方 keys](https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz) 提供。LA 含 TTS/VC 经 telephony/VoIP 编解码传播。 | 官网声明 LA/PA/DF 数据库为 **Open Data Commons Attribution Licence**。仅作冻结的外部语音留出，不能与 ASVspoof 2019 映射到的原始话语混入训练；官方也提供 [2019/2021 文件映射](https://www.asvspoof.org/asvspoof2019/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz) 用于去重。 |
| [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) | 58 位公众人物的网络真实/伪造语音；37.9 小时（20.7 real + 17.2 fake）。数据卡给出公开单 ZIP：`https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip`。 | 数据卡为 **CC BY-SA 4.0**，同时应保留人物、原始网页与采集限制记录；它是与实验室数据分布不同的外部语音 holdout，不训练、不调阈值。以 identity 为 group，防止同一人跨分区。 |
| [WaveFake](https://zenodo.org/records/5642694) | 104,885 个 fake WAV（LJSpeech、JSUT 与独立 TTS 管线），6 个 vocoder/生成路径；全量 `generated_audio.zip` 为 28.9 GB，真实参考音频**不随包分发**。 | **CC BY-SA 4.0**；LJSpeech 为 public domain，但 JSUT/其他参考并非同一许可，且 LJSpeech 为单说话人。全量 fake 加真实参考会挤占首批预算，故不作为首批；后续只在可按生成器和原始语料拆分、并与同源真实数据配对后使用。 |
| [MLAAD](https://deepfake-demo.aisec.fraunhofer.de/mlaad) | 50+ 语言、175+ TTS、约 1,000 小时 fake；真实音频另在 M-AILABS。 | **CC BY-NC 4.0**；超过预算且受非商业条款，不进入首批提交训练。 |

首批语音下载是 **ASVspoof 2019 LA 官方档案（7.12 GB）**；校验 MD5 后只保留
`train`、`dev` 与 protocols，`eval` 不能进入开发调参。ASVspoof 2021 先取 manifest/keys，
作为冻结留出待容量复核后再抓取 7.8 GB 媒体，避免压缩/解压缓存并存而越过 20 GiB 缓存
约束。若需逐 split 获取而非官方 ZIP，非官方
[Bisher/ASVspoof_2019_LA](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA) 镜像列出
train 25,380 / 1,515,101,755.52 bytes、validation（官方 dev）24,844 /
1,462,206,958.528 bytes，字段为 `speaker_id`、`audio_file_name`、`system_id`、`key`
（0 bona fide、1 spoof）；mirror commit、parquet shard hash 和官方 protocol 必须同时登记。
WaveFake 与 MLAAD 不能被其“公开”状态替代这一多说话人、官方 protocol 的路径。

## 视频：可访问候选与当前许可结论

| 候选 | 公开内容、标签、规模 | 许可/权利结论 | 当前动作 |
| --- | --- | --- | --- |
| [xmulz/AIGVDet](https://huggingface.co/datasets/xmulz/AIGVDet) | 13.2 GB；`train`、`val` 各有 `0_real`/`1_fake`，每条视频已抽帧；原论文 GVD 含 T2V 与 I2V 生成视频，非换脸专用 | 作者明确为 **academic research only**，禁止商业使用，并要求尊重原作者条款 | 容量满足首批预算，但“竞赛是否属于该许可允许的研究用途”及赛方外部数据限制均未证实；暂不下载、不训练。若两项获确认，需先列出每个视频的生成器/真实来源；当前目录结构没有足够的逐样本生成器 metadata。 |
| [Video Dataset: Perceptual Judgments of Authenticity](https://doi.org/10.5281/zenodo.19463142) | 816.0 MB；24 个场景，每场景 1 个 Pixabay 真实视频与 4 个 T2V 假视频（Veo 3.1、Seedance Pro 1、Sora 2、Kling 2.6）；`metadata.csv` 提供真实下载 URL、prompt 与四个生成文件名，包含人类与非人类运动 | Zenodo 公共记录的 Rights/License 文本未显示具体许可证；且真实视频只有链接、权利回到 Pixabay 条目 | 很适合小型、语义配对的外部 holdout，但在逐条确认 Rights 与 Pixabay 条目之前不能下载或训练。 |
| [OmerXYZ/comgenvid](https://huggingface.co/datasets/OmerXYZ/comgenvid) | 5,100 个视频，MSVD real 1,700、Sora fake 1,700、VEO3 fake 1,700；含 `subset`、`source_model`、路径和预计算嵌入；35.1 GB 完整媒体 | 卡片未声明数据许可；Sora 段说明权利属于原作者/OpenAI | 完整集超首批 30 GiB，且权利未清晰；不下载。预计算嵌入不能替代视频媒体训练集。 |
| [AIGVDBench](https://huggingface.co/datasets/AIGVDBench/AIGVDBench) | 440k+ 视频、31 个生成模型，含 T2V/I2V/V2V；公开 split 清单可作未来 generator holdout 设计 | 卡片标 CC BY 4.0，但总量远超项目预算，且每个来源和实际媒体文件仍须核验 | 仅作为未来抽样来源；不得全量镜像。 |
| [ViF-CoT-4K](https://huggingface.co/datasets/JoeLeelyf/ViF-CoT-4K) | 固定 `c9e9b30e47e216f72c50664c43750674a79c0ded` 的 `source_videos.zip` 为 1,294,907,616 B；Range 读取证实内置 `normalized.json` 有 3,502 对、7 个 CogVideoX/Hunyuan/Wan 生成器路径，来源为 Panda-70M 2,306 对、kinetics 1,196 对。18.36 GB 的 parsed_frames 不下载。 | Skyra 声明 **CC BY 4.0** 且要求遵守上游条款。[Panda/HD-VILA 原文](source/vif-panda/provenance.json)允许限定的本地非商业计算研究；竞赛用途仍未确认。Kinetics 媒体的研究授权未闭环，当前排除。 | 只准备 Panda 研究子集；核验完整 ZIP SHA-256、来源条款、已知配对及保守疑似关系约束后再解码/探索训练。未确认的原录制与身份保留 unknown，不报告独立组 CI 或完整阶段验收。另一个 Skyra 细粒度标注文件存在路径版本差异，不能直接 join。 |

## 执行前闸门

1. 分别登记本地研究用途和竞赛用途。来源条款已明确允许的本地非商业研究可在该范围
   训练，数据与产物须标记 `local_research_only`；许可未解决时仍停止相关训练。
   外部数据、非商业/研究许可及预训练权重的赛用细则未确认前，有条件资源不得进入
   正式提交候选或被登记为竞赛已批准。该区分用于继续独立研究，不扩大上游许可。
2. 对 EnvSDD，先完成上游真实源许可和 group 关系抽查；下载选择必须由固定 manifest
   驱动，manifest 自身也记录 SHA-256。
3. 对视频，Panda 研究子集的二元索引已定位；许可原文、归属、媒体哈希、已知配对和
   疑似关系约束须随清单登记。未知原录制 ID 不捏造，允许明确受限的探索性训练，
   但不能报告统计独立性、正式晋级或完整阶段验收。关键帧近重复及编码捷径仍须检查。
   当前没有已验收的正式赛用视频集，
   不能将研究许可或人脸换脸集冒充通用生成视频赛用验收。
4. 每次下载后立即用 `ffprobe` 抽查 codec、时长、音轨、分辨率和可解码性；对真实/伪造
   对称重编码。解码失败保留在 manifest，不能静默删除。

### 公共只读元数据接口

在下载任何大文件前，优先调用以下 Hugging Face Dataset Viewer GET 接口；`config` 与
真实 split 必须以 `/splits` 响应为准，而不是数据卡文字。

```text
https://datasets-server.huggingface.co/is-valid?dataset=EnvSDD%2FEnvSDD
https://datasets-server.huggingface.co/splits?dataset=EnvSDD%2FEnvSDD
https://datasets-server.huggingface.co/rows?dataset=EnvSDD%2FEnvSDD&config=default&split=train&offset=0&length=100
https://datasets-server.huggingface.co/parquet?dataset=EnvSDD%2FEnvSDD
https://datasets-server.huggingface.co/size?dataset=EnvSDD%2FEnvSDD
https://huggingface.co/api/datasets/EnvSDD/EnvSDD/tree/main/data?recursive=false&expand=false
https://huggingface.co/api/datasets/JoeLeelyf/ViF-CoT-4K/tree/main?recursive=true&expand=false
```

## 不作为首批训练来源的常见误判

- AIGVDet 公开且仅 13.2 GB，不等于可以绕过其 academic-only 条款。
- FakeSound 是 Apache-2.0 且仅约 987 MB，不等于包含 bona fide 训练标签。
- ComGenVid 具有明确 `source_model`，但其完整媒体为 35.1 GB、Sora 段权利保留；不适合
  当前本地预算。
- Zenodo 页面可下载，不等于 Rights 字段为空时已获得再利用许可。
- ASVspoof 等语音语料可用于语音反欺骗对照，却不能证明覆盖赛题可能包含的音乐、音效、
  环境声或混合音频；它们不替代本记录的非语音数据闸门。
