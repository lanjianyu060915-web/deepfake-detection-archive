# 固定 CLIP＋WaveRep 离线视频融合

目标：将 #59 的互补分数接入现有 `infer` / `predict`，完成 #72 的研究候选，
不再围绕帧数和小型分类头做无收益扫描。使用冻结参数，不训练、不重新拟合、不读取 holdout。
完整数值与资源见 [summary.json](summary.json)，实值配置见 [infer-controlled.json](infer-controlled.json)。

## 方法与数据边界

CLIP 使用已选定的 seed43 MLP，4 帧 224 原生中心裁剪；WaveRep G1 使用 4 帧
504 原生中心裁剪。两路保留各自 RGB 归一化，同一媒体、时间点与选定视频流。
各自先平均原始帧 logit，再按 `sum(coef * (logit - mean) / std) + intercept`
输出一个连续、高分更伪造的融合 logit。没有 sigmoid、单分支 fallback 或失败补零。

`mean/std/coef/intercept` 来自 #70 的 `clip43_waverep_fusion`，仅在原 dev844 拟合；
数值随两路完整权重和 CLIP MLP 一起写入 `video_fusion_v1` checkpoint。
权重、构造尺寸、帧数和标定由 checkpoint SHA 绑定；归一化与裁剪实现由既有源码身份绑定。
组合身份进入现有恢复机制，资源只在入口加载与校验，逐视频不重新哈希模型权重。

本轮完整评估 AIGVDBench 受控 dev1880：600 真、600 配对 Open-Sora、另17生成器各40假。
媒体对称转为 448×256、16fps、1.25秒、H.264 CRF23。`Opensora` 的40条与
`Open-Sora` 的600条是不同组，去 Open-Sora 时保留前者。
CLIP 对照复用已完成的标准 `infer` 分数；WaveRep 研究分数仅用于解释路径差异，
不会冒充标准入口分数。融合候选的1880条均实际重新从媒体经过公开评分器。

该 dev 已用于多轮研究，不是盲测。旧 Panda 母媒体关系仍不全，不能证明与新 YouTube
来源零交集；WaveRep G1 训练过 Pyramid-Flow，本卡另报去 Pyramid-Flow 结果，
不称全部生成器未见。数据底层 YouTube 许可与 WaveRep 正式赛用均未确认。

## 完整结果

以下为 EER（%）。标准融合、标准 CLIP 对照与重建研究融合均对齐同1880个样本 ID 和标签，
全部成功、覆盖率1.0。研究融合用既存分数和冻结参数重建，没有重新拟合。

| 视图 | 样本数 | 标准 CLIP43 | 标准融合 | 研究融合 |
| --- | ---: | ---: | ---: | ---: |
| pooled | 1880 | 31.333333 | **26.406250** | 26.484375 |
| 去 Open-Sora | 1280 | 29.852941 | **21.323529** | 21.323529 |
| 去 Pyramid-Flow | 1840 | 31.854839 | **27.000000** | 27.016129 |
| 配对 Open-Sora＋600真 | 1200 | 32.666667 | **31.333333** | 31.333333 |

pooled 改善4.927083个百分点；18个生成器各自对全部600真样本的 EER 均低于 CLIP 对照，
但 Open-Sora、SEINE 仍然困难，不能据此认为视频泛化已经解决。

| 生成器 | 标准 CLIP43 | 标准融合 |
| --- | ---: | ---: |
| AccVideo | 25.67 | 19.00 |
| AnimateDiff | 22.50 | 18.17 |
| EasyAnimate | 32.50 | 22.83 |
| Gen2 | 32.50 | 19.33 |
| Gen3 | 35.00 | 25.83 |
| IPOC | 20.67 | 16.00 |
| Jimeng | 30.00 | 22.50 |
| Luma | 42.50 | 25.50 |
| Open-Sora | 32.67 | 31.33 |
| Opensora | 33.67 | 15.00 |
| Pyramid-Flow | 25.00 | 12.50 |
| RepVideo | 32.50 | 25.00 |
| SEINE | 42.50 | 40.00 |
| SVD | 35.00 | 24.17 |
| Sora | 31.33 | 22.50 |
| VideoCrafter | 22.17 | 12.50 |
| causvid_24fps | 26.83 | 7.83 |
| vidu | 35.00 | 27.50 |

标准与研究融合的最大/平均绝对分差为0.049366/0.003608，pooled EER 相差−0.078125个百分点。
两者使用相同权重和固定系数。抽查分差最大的三条，WaveRep 的研究/标准输入逐值一致，
同一模型重算也接近当前标准分数；历史 CLIP 分数却与当前分支相差约0.08–0.114。
单独启用 cuDNN TF32 没有重现旧值，尚未完全隔离执行差异，不能归因于解码或某个精度开关。
本轮保留实际分数与三条诊断，不为这0.078125个百分点重跑全矩阵或宣称额外算法收益。

完整1880条的标准 CLI 调用803.54秒，包含清单预检、模型加载和评分，
不包含 Python/Torch 导入及 CUDA 初始化。allocated 峰值1,801,065,472字节，
reserved峰值1,935,671,296字节；进程RSS峰值3,998,792KiB。
这是本次实际测量，不据单次共享负载比较宣称吞吐提升或投影比赛时限。

## 标准入口与资源

准备工具只组装三份已固定 SHA 的本地资源，不拟合参数：

```bash
scripts/devctl run python tools/prepare_video_fusion.py \
  --clip /var/tmp/deepfake-video-ufd-20261014/models/ufd-vit-l14.pt \
  --clip-head /var/tmp/deepfake-video-clip-head-20261014/seed43/training/best.pt \
  --waverep /mnt/deepfake-archive/issue59-generator-forensics/models/waverep-g1.pt \
  --output models/video-fusion.pt
cp configs/infer-video-fusion.example.json configs/infer-video-fusion.local.json
# 填入实际 dev 清单、媒体根目录、checkpoint 和新的输出目录
scripts/devctl run python -m deepfake_detection_archive infer \
  --config configs/infer-video-fusion.local.json
```

checkpoint SHA：`068a079949e0e5764c649986ebe7ae6e2eaea56a9329101da380b9cc77130f31`。
运行源码固定为 `ed18da6`，FP32、matmul/cuDNN TF32 关闭，主干帧 batch4。
两路各自解码以保留裁剪，不共用缩放后的输入。没有改动训练器、采样数、发布默认或旧缓存。

本轮直接复用 `Builder.copy_weight` 收集单 checkpoint 与 sidecar，配以相对路径配置，
无需增加融合专用发布框架。非 editable wheel 从源码外中文/空格 CWD、空 HF 缓存和
只读搬移模型包运行，原发布 AST 与新视频融合共用 `predict`，2音频＋2视频全部成功。
依赖来自已安装 cu132 环境；这是小批集成结果，不是新环境完整发行、EER或万级容量证明。

运行资产：`/var/tmp/deepfake-video-fusion-20261016/`；最小持久集合：
`/mnt/deepfake-archive/C8-video-fusion-20261016/`。保留权重、配置、逐样本分数、
摘要、必要资源记录与 wheel，不备份媒体全集、特征缓存或重复运行环境。

## 采用边界

仅保留研究候选。正式赛用许可、冻结后保留集验收和最终发行容量仍待完成，
不将 WaveRep 权重加入正式 Release，不替换 v0.1.1 默认。
后续优先 #51 的真实来源、困难内容和可训练数据，避免继续据该已见 dev 调系数。
