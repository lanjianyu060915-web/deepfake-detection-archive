# 项目状态

更新：2026-10-17。**v0.1.1 是可运行的环境音/视频离线研究基线**：
带本地权重、CPython 与依赖，无标签目录→推理→连续分数导出，支持失败记账与恢复。
[下载与使用](release.md)。三域联合音频研究候选现已完成，尚未替换发布包；
新RC3已完成音频/视频各万条零失败容量运行；独立来源泛化、官方格式、
赛用许可与GPU容器实测仍有缺口。
后续见[路线图](roadmap.md)，全部实验见[实验索引](experiments/README.md)，资产见[数据说明](data.md)。

## 当前候选与缺口

| 方向 | 已有结果 | 当前选择与缺口 |
| --- | --- | --- |
| 语音 | 带宽seed42/43 clean 0.90%/1.08%、8kHz带限1.39%/1.04%；In-the-Wild 30.67%/31.19% | 保留带宽seed42；白噪声双种子不稳定已停止；真实网络语音仍弱；[实验](experiments/speech-robust-20261013/README.md) |
| 语音SSL | 官方XLS-R+AASIST LA权重：ASV dev clean/AAC/8kHz 0.08%/0.08%/0.04%，A07–A19 0.19%，In-the-Wild 11.21%（LogMel 30.67%、AST 24.86%） | 保留为语音专家候选，不替换通用音频；权重1.2GB、显存约2GB；[实验](experiments/speech-ssl-20261014/README.md) |
| 环境音 | Urban AST12层seed42 dev1.39%；冻结研究候选holdout已见1.43%、未见生成器6.51% | 保留环境音对照；不能代表音乐或通用音频；[实验](experiments/audio-envsdd-20261011/README.md) |
| 音乐 | 扩展AIME无增强候选未训练路径18.75%；事后等权分数融合14.58% | 融合仅为dev探索，未接离线推理；跨来源仍弱；[实验](experiments/audio-ast-regularization-20261012/README.md) |
| 音乐/语音混合 | 匹配对照中未训练路径合成混合32.55%→26.89%；真实广播混合仍50%、环境音退化 | 保留负结果，不替换主候选；[实验](experiments/audio-speech-mix-20261012/README.md) |
| 三域联合音频 | 等权双种子dev macro 0.70%/0.81%、pooled 1.14%/0.87%；主候选冻结holdout fsd8.57%、音乐2.08%；In-the-Wild24.86%；未训练音乐25.00%、广播混合51.32% | 冻结seed42为主、seed43备选，权重/配置/完整分数团队可读；泛化与许可尚不足以替换release；[实验](experiments/universal-audio-20261012/README.md) |
| 视频 | 七生成器train3,614/dev844，四轮训练；统一1949条EER33.12%→31.73%，未训练标签33.47%→33.14% | 原同域dev16.44%→19.18%，保留研究候选、不替换v0.1.1；仍缺独立来源泛化与更强视觉表征；[实验](experiments/video-multigen-20261013/README.md) |
| WaveRep与固定研究融合 | G1新来源受控dev31.41%；仅原dev844拟合的CLIP43+WaveRep为26.48% | 原研究路径结果；标准融合入口及其全量结果见下方，冻结保留与许可待确认；[PR #70](https://github.com/lanjianyu060915-web/deepfake-detection-archive/pull/70) |
| 强视频公开权重对照 | UFD完整1949条47.75%、未训练标签50.00%、旧dev49.32%、CRF35 58.43%；共2405条全覆盖、零失败 | 作者数值已对齐；停止直接采用公开检测头，不替换现候选，继续有限适配与独立来源比较；[实验](experiments/video-ufd-20261014/README.md) |
| 冻结CLIP视频头 | 双种子dev24.88%/24.64%；冻结seed43统一1949条26.56%、未训练标签27.88%、CRF35 25.84% | 优于七生成器ResNet，原同域dev20.55%仍退化；新来源受控dev1880条31.33%（ResNet39.00%）、自然组21.33%（ResNet43.36%），8帧31.80%无收益、保留4帧，保留为原冻结对照，跨来源适配见下方；母媒体独立性及许可未确认，不替换release；[新来源实验](experiments/video-aigvdbench-20261014/README.md) |
| 冻结CLIP轻量时序头 | 双seed原dev23.93%/24.41%，选42；新来源受控31.17%，去Open-Sora30.29% | 较MLP43 pooled仅改善0.16点、去Open-Sora退化0.44点；不替换候选，停止轻头/帧数调参；[实验](experiments/video-clip-temporal-20261014/README.md) |
| CLIP跨来源训练（#77） | 新增officialtrain的600train/200dev，与旧域回放；主43外部受控dev31.33%→21.25%、复现42为21.00%；去本轮新增两标签29.22%→24.00%/23.33% | 前轮研究对照，后续主候选见下一行；旧域24.64%→25.83%略退；同AIGVD已见dev、非盲测，许可及最终保留未完成，不替换release；[实验](experiments/video-cross-source-20261016/README.md) |
| CLIP三生成器扩展（#80） | 新增1500条库存，排除跨split重复原组2条后train5412/dev1344；主43外部AIGVD18.83%（上轮21.25%、同预算control21.67%）、复现20.00%；ViF pooled25.91%（上轮27.75%） | 采用主43为后续训练研究候选；旧域回到24.88%，原新dev200由16%→18%、ViF未训练标签29.21%仍差于原CLIP27.88%；已见诊断非盲测，许可/最终保留未完成；[实验](experiments/video-generator-coverage-20261016/README.md) |
| CLIP最后block适配（#81） | 固定主43 AIGVD18.83%→17.83%、去五训练标签21.00%→19.33%，旧dev24.88%→23.22%；ViF pooled25.96%、未训练标签28.64%，rep42为26.92%/30.36% | 保留微调研究候选，C10仍通用视频主候选；复现ViF退化，不升级Release或继续扫已见dev；完整权重已存C11，主43标准infer4/4；[实验](experiments/video-clip-adapt-20261016/README.md) |
| CLIP/DINOv2同预算训练（#86） | 八组训练完成；新CLIP43/42 AIGVD15.86%/17.00%（C10 18.83%），ViF26.67%/26.92%（C10 25.91%）；DINO适配主43为24.53%/33.65% | C10仍为通用研究主候选，新CLIP为研究备选，停止本轮DINO训练；已见诊断非盲测，保持发布默认；[实验](experiments/video-dinov2-20261017/README.md) |
| 原生Swin3D与CLIP16（#88） | 八组同预算训练完成；Swin适配主43 AIGVD20.94%/ViF34.58%，匹配CLIP16为16.83%/26.12%；时序置乱256条已完成 | Swin有时序敏感性但弱于CLIP，停止本轮投入；CLIP16较4帧跨集有取舍，保留对照，C10仍主候选；[实验](experiments/video-native-20261017/README.md) |
| 三生成器训练覆盖扩展（#90） | 新增2400条officialtrain，实际train7812/dev1344；四组同预算训练，主43 AIGVD17.11%、去八训练标签20.33%，旧dev24.88%；ViF26.92% | AIGVD优于C10及匹配control，ViF却差于C10的25.91%，保留研究备选，C10仍主候选；下一轮优先中文平台数据，停止本轮同体系扩展；[实验](experiments/video-coverage-20261017/README.md) |
| 固定CLIP＋WaveRep融合 | 真实标准入口受控dev1880/1880、零失败；EER26.41%（CLIP31.33%）、去Open-Sora21.32%；18个生成器均优于CLIP；CLI调用803.54秒、显存allocated峰值1.80GB | 已接入infer/predict、单checkpoint与搬移wheel；仅研究候选，dev已见、旧母媒体交集未知，冻结保留验收及赛用许可待确认；[实验](experiments/video-fusion-20261016/README.md) |
| cu126与固定工具 | 完整cu126三件套实装；RC3原权重20输入在CPU/GPU/Ubuntu24新用户各20/20，30前端张量差0、CPU/GPU分数差≤7.63e-6；同设备旧/新实现分数差0 | 系统Toolkit13.2保留；固定同源码LGPL工具及对应源码随包；Ubuntu24用户态共享宿主内核，不能当作Docker；[本轮回归](experiments/runtime-completion-20261016/README.md) |
| 离线系统 | clean `2e8388a` RC3、adaptive模式：音频10000/10000、1046.83秒；视频10000/10000、1613.32秒，均零失败；allocated/reserved峰值分别400.12/434.11MB、110.13/134.22MB | 两任务并行，耗时不与旧串行运行作加速比较；已见诊断/train-dev容量，无holdout或EER，默认仍为原v0.1.1研究权重；CPU/cu126无权重Docker镜像已实测CPU推理及save/load，未重发Release；[结果与限制](experiments/runtime-completion-20261016/README.md) |

以上不是比赛成绩。music-aime96、speech-asv200来自当前dev；未训练音乐视图共享既有dev
真样本与prompt组，原曲/artist独立性未知。AI-OpenBMAT有标签纯度与底层许可限制。
In-the-Wild及多轮使用的dev已见，不能再作为未来盲测；正式交付仍须满足[比赛要求](competition/requirements.md)。

## 候选状态（2026-10-17 核对）

按产物核对，不以旧评论或 `running` 字段为准；权重哈希与持久副本见[数据说明](data.md#持久归档)。
“复核”指他人重跑结果，不等于独立来源泛化。#51已交首批视频dev，受控组四候选完整比较已完成；
与旧Panda训练/选择集的YouTube母媒体交集未知，最终保留评价尚未进行。

| 候选 | 训练/评分 | 选定 | 复核 | 合并 | 发布 | 权重可恢复 | 标准入口加载 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.1.1 环境音AST + 视频ResNet18 | 完成 | 发布默认 | release复核、CPU/GPU演示 | #42、#44 | v0.1.1 | 是 | 发布包可用 |
| 三域联合AST seed42主/seed43备 | 完成 | 研究主候选 | #56复核 | #45、#56 | 否 | 是 | 发布包`infer`可加载，归档副本80条分数逐位一致 |
| LogMel带宽 seed42冻结/seed43 | 完成 | 语音对照 | seed43经#55复核，seed42仅自评 | #37、#55 | 否 | 是 | 仓库源码入口 |
| 七生成器视频 paired-seed43 | 完成 | 研究候选 | #48复核 | #48 | 否 | 是 | 发布包不兼容（`temporal_kernel_size`），仓库源码入口20条最大差4.9e-3 |
| 视频时序头 seed43 | 完成 | 研究候选 | 复核目录团队不可读 | #43 | 否 | 是（原权重root 0600，归档副本可读） | 未演练 |
| 冻结CLIP视频头 seed43 | 完成 | 原冻结对照 | 未独立重跑完整集 | #61、#64 | 否 | 是（C6） | 标准infer可加载；C6两条恢复分数一致 |
| CLIP跨来源训练 seed43主/seed42复现 | 完成 | 前轮研究对照 | 产物/统计独立核对，未独立重训 | #79 | 否 | 是（C9） | 持久权重标准infer 4/4加载，完整外部诊断为共同特征评分 |
| CLIP三生成器扩展 expanded43主/42复现 | 完成，含同预算control43 | 后续训练研究主候选 | 产物/统计独立核对，未独立重训 | #79 | 否 | 是（C10） | 持久权重标准infer 4/4加载，完整结果为共同特征诊断 |
| CLIP最后block tuned43主/42复现 | 完成，含匹配control43 | 研究保留，通用主候选仍C10 | 训练预算、代表参数及188项EER/计数独立核对，未独立重训 | #82 | 否 | 是（C11） | 主43完整checkpoint标准infer4/4，最大logit差4.65e-6；全量外部为共同前缀诊断 |
| 新初始化CLIP43主/42复现 | 完成，同数据/预算的新头 | 研究备选，C10仍主候选 | 八组预算及54项EER/计数独立核对，未独立重训 | #87已合入main | 否 | 是（C17），当前实例共享只读 | 两头标准原视频各620/620成功，CRF18/35同310ID；完整外部为特征诊断 |
| DINOv2冻结43/42与末block适配/对照 | 完成，八组比较中的六组 | 负结果，停止本轮投入 | 同上；区分FP32表征与BF16前缀 | #87已合入main | 否 | 是（C17/C18），当前实例共享只读 | 冻结43/适配43/对照43各620/620；其余为特征或前缀诊断 |
| 原生Swin3D冻结/适配/对照与CLIP16 | 八组训练及完整诊断完成 | Swin负结果、CLIP16研究对照，C10仍主候选 | 预算/分数与时序诊断独立核对，未独立重训 | #89已合入main | 否 | 是（C19），当前实例共享只读 | 冻结Swin43/适配43/CLIP16_43原视频各8/8，最大缓存差2.41e-5；时序9组2304/2304 |
| 三生成器覆盖 expanded43/42与control43/42 | 四组20epoch、1240updates全部完成 | 跨集取舍，研究备选，C10仍主候选 | 结果与产物独立核对，未独立重训 | #94 | 否 | 是（C22），团队共享只读 | 主43标准CLI原视频8/8，最大缓存logit差7.92e-5；完整外部为缓存诊断 |
| CLIP43＋WaveRep固定融合 | 标准dev1880完成 | 研究候选 | 实验记录独立核对；未独立重跑 | #76及补集成#78已合入main | 否 | 是（C8） | infer全量、非editable wheel统一入口2音频＋2视频成功 |

所有候选的正式赛用许可均未确认。三域AST两种子与时序头为脏工作区产物，LogMel seed43未记录commit
（实际为#55分支`2fabb61`），代码以 `code_sha256` 追溯。

## 后续主线

#39 的第二种子/噪声对照、#40 的多生成器训练、#41 的双种子联合训练及冻结诊断已交付。
#39 已按负责人要求收尾，来源许可不作为该 Issue 的完成条件。
新的比赛主线由 [#49](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/49) 汇总，
具体认领、进程和PR在Issue维护：

- 独立来源数据与评测隔离：[#51](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/51)。
- 更强视频基线：[#52](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/52)；语音SSL对照：[#53](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/53)。
- 统一媒体预测入口：[#57](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/57)已完成，经#62/#64合入main；实际交付由#54继续。
- 视频固定融合：[#72](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/72)标准入口已完成；后续优先独立数据与困难内容，不据已见dev继续调系数。CPU/cu126安装与固定帧分批分别由#73/#74、#75交付，#74/#75/#76经#78补集成到main；本轮纠正#74的CPU TorchAudio构建，完整cu126三件套已实装并通过GPU算子与原权重回归。
- 跨来源视频训练：[#77](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/77)完成新数据、双种子适配与冻结诊断；优先扩充困难内容及真实来源，不继续扫描已见dev。
- 视频训练覆盖：[#80](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/80)完成AccVideo/EasyAnimate/SVD新训练、同预算对照及跨集诊断；继续更多来源/困难内容与强表征有限适配，保留旧域能力。
- 视频表征有限适配：[#81](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/81)完成最后block双种子与同预算头对照；AIGVD及旧域改善，ViF复现退化，研究保留而不替换C10。下一步更多真实来源/困难内容，与#51并行衔接，不继续扫已见dev。
- 新表征匹配训练：[#86](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/86)完成原始DINOv2/CLIP双种子及DINO末block/头对照；新CLIP改善AIGVD、ViF仍有取舍，保留研究备选；停止本轮DINO训练，后续优先新真实来源、困难内容和视频时序表征。
- 原生视频时序：[#88](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/88)完成Swin3D/CLIP16匹配双种子与Swin末stage/头对照，完整外部和时序诊断均齐备；Swin仍弱于CLIP，停止本轮训练，CLIP16保留对照。后续优先#51新真实来源/困难生成内容，不继续扫已见dev。
- 新增三生成器训练：[#90](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/90)完成2400条扩展与四组匹配训练；AIGVD改善、ViF退化，保留研究备选。下一轮承接#51中文平台数据的分组划分，不继续仅扩AIGVD或扫已见dev。
- 安装与资源适配：#65/#66的实现和实际RC3回归已通过[PR #85](https://github.com/lanjianyu060915-web/deepfake-detection-archive/pull/85)合入main；默认reference、显式adaptive，有界profile/OOM缩减与恢复计划已接入。Docker CPU/cu126镜像在无GPU Actions以CPU模式做4样本及save/load演练，[运行](https://github.com/lanjianyu060915-web/deepfake-detection-archive/actions/runs/38916789385)已成功，两路4/4且host/container/save-load分数差0；不计GPU容器或万级容器验收。
- 共享资产持久保存：[#50](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/50)原备份完成；用户后来增加的恢复要求尚未全部完成，已重新开放，C14保存部分恢复结果；离线兼容与容量：[#54](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/54)。

RTX4090按实际负载共享，任务输出独立。联合音频候选位于
`/srv/shared-data/candidates/audio-universal-20261014/`，文件只读、团队可读取；
该路径仍为当前实例共享副本；最小候选集合已另存[持久归档](data.md#持久归档)，
跨实例恢复尚未演练。日常开发无需重建环境或安装记录。
