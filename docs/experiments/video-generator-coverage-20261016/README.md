# 视频训练覆盖扩展与匹配预算对照

固定主43在AIGVDBench已见受控dev EER为**18.8281%**，上一轮21.25%、同预算对照21.6667%；
复现42为20.00%。历史ViF pooled为25.9079%，优于上一轮27.7507%及原CLIP26.5583%。
共同dev source macro也优于同预算对照，**采用主43作为后续训练研究候选**，不改选seed42。
原域小幅退化大部分恢复，但原新域dev200由16%退到18%、ViF未训练标签仍差于原CLIP，
不能声称全面泛化改善；Release默认与旧固定融合保持不变。完整数值见[摘要](summary.json)。

## 数据与方法

承接[#77 / PR #79](https://github.com/lanjianyu060915-web/deepfake-detection-archive/pull/79)，
扩展[AIGVDBench固定版本](https://huggingface.co/datasets/AIGVDBench/AIGVDBench/tree/bf48acaa4920990af3dd2a511ea88e63354df305)
中确实具有officialtrain的AccVideo、EasyAnimate T2V、SVD I2V。
[官方划分](https://github.com/LongMa-2025/AIGVDBench/blob/e38c75abda3d319c0c3072c77594f6eeef1c028f/README.md#-dataset-split)
明确闭源生成器仅测试；不因Gen3/Jimeng/Luma等dev较弱就把其测试媒体用于训练。

沿用#77已知5870个YouTube排除组，再排除其400个既有组。
剩余组按`sha256(43:YouTubeID)`排序再取600个train组，同组clip按
`sha256(43:filename)`选定，每组一real加一个fake，三个新增生成器交替、各200条。
原100个dev组各补三个fake共300条，不重复下载real、不改变原划分。

| 本轮新增 | 原组 | real | AccVideo | EasyAnimate | SVD | 总数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 600个全新组 | 600 | 200 | 200 | 200 | 1200 |
| dev | 原100组 | 0 | 100 | 100 | 100 | 300 |

计划与#77合并后train5414、dev1344。实际准备发现AccVideo的不同YouTube ID
`K0XTykJIsZ0`（train）和`c0bnCKKuPlM`（dev）原媒体字节相同，规范化后也相同。
在任何扩展梯度训练前保留dev，移除训练原组的real/fake两条，不补选。
最终新增训练1198条/599组，总train**5412**（旧Panda3614、新AIGVD1798）、
dev1344（旧844、新500）。原1500条获取库存保留，训练使用1498条新记录；
证据与两个排除ID见C10的`training-exclusions.json`，原采样plan未改。
新域dev为100real/400fake，保留原计数；各生成器EER共享真实池，不相加为独立实验。
新原媒体使用逐ZIP成员有界Range，不整包下载；局部header、filename、CRC、解压长度及媒体信息
由原下载器处理。本轮仅以小wrapper合并header/payload请求并使用12并发，采样计划不变。

真假对称统一448×256、16fps、1.25秒、H264 CRF23，再取4帧224 native center crop；
旧Panda的2秒媒体及旧缓存不重提，保留时长差异。已知组交集为0；
未知Panda原YouTube身份仍未知，不能宣称跨数据集母媒体完全独立。
原媒体只作内部研究，发布者CC-BY-4.0声明不代表底层视频权利与正式赛用已解决。

## 匹配预算与选择

三个训练均初始化原CLIP43 MLP头，冻结同一CLIP ViT-L/14骨干，隐藏64、dropout0.1。
control43只用#77已有4214条train，expanded43/42用扩展5412条train；
三者使用完全相同dev1344，原844/新500的source macro EER选择epoch。
primary expanded43预先固定，42仅复现，control43仅比较；外部诊断不选择epoch或种子。

source×label等权有放回抽样，每epoch固定5414次、batch128，保留最后短batch；
20完整epoch、860次optimizer更新、108280次抽样；patience20保证预算完整。
AdamW lr1e-3/wd1e-4、BF16、grad norm1，OMP/MKL=2。
仅实际选中的epoch可能不同；数据扩展同时改变同预算下的样本重复率，不能将结果拆成单一机制。

实际入口为C10中的`run_training.py --control-only`及无参数全量阶段，复用同一trainer、
_CachedDataset与_balanced_sampler；新增dev先就绪后训练control，全量阶段直接复用冻结control。
原4455个Panda缓存、前轮800个新域缓存只在各进程进入训练前验证一次，身份不改写。
仅提本轮实际使用1498条及历史ViF1949条特征，不重提AIGVD1880或旧训练缓存。
训练固定源码`35de6a61b43ff895cf911ab55fa69b445e36407a`（clean），无新增依赖/训练框架。
因重复组排除而调整的两行适配器差异保留为`adapter-exclusion.patch`，可恢复control使用的原脚本；
训练预算及共同dev未改，已完成control未重复训练。

## 冻结诊断与适用范围

三种训练完成并冻结checkpoint后，五个头在同一特征上评分：原CLIP43、#77主43、control43、
expanded43、expanded42。AIGVD1880直接复用前轮特征，ViF受控1949本轮只提一次。
两者都是已见诊断集，不是最终盲测。AIGVD五个训练标签为SEINE、Open-Sora、AccVideo、
EasyAnimate、SVD；去掉后剩13标签，仍属同一数据域且方法家族可能重叠。
ViF未训练标签也只指清单标签未进原训练，不保证家族或母媒体独立。

本轮不读取holdout媒体/分数，不套用或重拟合旧CLIP＋WaveRep校准，不替换Release默认。
特征缓存诊断的耗时不代表公共CLI端到端容量；实际覆盖、成本、退化及采用/停止结论见本轮摘要。

## 完整比较

所有五个头分别1344/1344、1880/1880、1949/1949，零失败。下表均为百分比，低更好；
原CLIP43和上一轮77两种冻结基线也在本轮共同特征上评分，不混用不同执行条件的分数。

| 视图 | real/fake | 原CLIP43 | 上轮77 | control43 | 主expanded43 | 复现expanded42 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原Panda dev844 | 422/422 | 24.6445% | 25.8294% | 25.5924% | 24.8815% | 24.4076% |
| 扩展新域dev500 | 100/400 | 28.2500% | 24.0000% | 20.0000% | 18.0000% | 18.0000% |
| 原新域dev200 | 100/100 | 34.0000% | 16.0000% | 19.0000% | 18.0000% | 17.0000% |
| 新增三生成器dev400 | 100/300 | 27.0000% | 25.0000% | 21.3333% | 18.0000% | 18.3333% |
| 共同dev pooled1344 | 522/822 | 24.9392% | 25.4789% | 25.2874% | 22.5061% | 22.0307% |
| AIGVD pooled1880 | 600/1280 | 31.3333% | 21.2500% | 21.6667% | 18.8281% | 20.0000% |
| AIGVD去Open-Sora1280 | 600/680 | 29.8529% | 24.2647% | 23.8333% | 21.3333% | 21.3333% |
| AIGVD去五训练标签1120 | 600/520 | 29.0000% | 23.4615% | 22.8333% | 21.0000% | 21.3333% |
| ViF pooled1949 | 104/1845 | 26.5583% | 27.7507% | 26.9919% | 25.9079% | 26.0163% |
| ViF未训练标签视图 | 104/1222 | 27.8846% | 29.9509% | 30.7692% | 29.2144% | 28.9689% |

原域24.6445%→24.8815%仍略退，较上一轮25.8294%接近恢复。ViF整体仅104条real对1845条fake，
未训练标签视图仍29.2144%对原CLIP27.8846%退化，不能以pooled改善掩盖这一缺口。
原新域dev200也由16%→18%，扩展新域dev500改善主要覆盖新增生成器，仍需保留原对照。
所有诊断已见且母媒体独立性未全面确认，小差异不作为稳定排名或比赛成绩。

| 训练 | 所选epoch | 共同source macro | head训练秒 |
| --- | ---: | ---: | ---: |
| control43 | 13 | 22.7962% | 23.66 |
| expanded43 | 14 | 21.4408% | 35.82 |
| expanded42 | 10 | 21.2038% | 23.34 |

三个训练实际均20epoch、860updates、108280draws、0skipped；权重先冻结再运行外部诊断。
对照说明单纯继续训练也会改变结果；新增覆盖相对同预算对照进一步改善，但依然只是有限配方对比。

AIGVD逐生成器均共享600条real：Open-Sora600fake，其余各40fake；各桶不是独立重复实验。

| 生成器 | 上轮77 | control43 | 主43 | 复现42 |
| --- | ---: | ---: | ---: | ---: |
| AccVideo | 26.00% | 27.33% | 15.50% | 17.50% |
| AnimateDiff | 16.83% | 17.50% | 12.50% | 15.00% |
| EasyAnimate | 31.17% | 34.50% | 20.33% | 22.50% |
| Gen2 | 29.00% | 25.33% | 25.17% | 24.50% |
| Gen3 | 29.17% | 28.83% | 24.83% | 26.50% |
| IPOC | 15.00% | 16.33% | 10.00% | 7.50% |
| Jimeng | 29.67% | 30.00% | 25.00% | 23.00% |
| Luma | 30.00% | 27.50% | 28.83% | 30.00% |
| Open-Sora | 18.00% | 19.33% | 16.17% | 17.50% |
| Opensora | 27.50% | 25.00% | 25.00% | 24.17% |
| Pyramid-Flow | 12.50% | 15.67% | 12.50% | 12.50% |
| RepVideo | 25.00% | 25.50% | 18.50% | 20.00% |
| SEINE | 27.50% | 25.00% | 28.00% | 27.83% |
| SVD | 20.00% | 22.33% | 17.50% | 20.00% |
| Sora | 27.50% | 27.50% | 24.17% | 25.00% |
| VideoCrafter | 15.33% | 12.50% | 15.00% | 15.00% |
| causvid_24fps | 11.83% | 12.50% | 10.50% | 12.00% |
| vidu | 23.00% | 25.00% | 25.50% | 25.00% |

AccVideo/EasyAnimate的明显改善发生在已经训练过这两个标签之后，不能称未见生成器提升。
SEINE和vidu相较上轮退化；Luma/VideoCrafter等相对control也有退化，保留全部桶而不只报告改善项。

## 资源与恢复

1500条原媒体3,010,025,217字节、受控媒体76,353,255字节，下载和预处理零失败；
从固定plan到下载完成1177.79秒，优化续跑820.7秒不代表全量总耗时。
新增dev先到后完成control阶段68.97秒；扩展阶段145.62秒，两阶段含特征准备及训练，
不含下载/转码与历史ViF特征提取。三个head训练分别23.66/35.82/23.34秒；
历史ViF1949特征120.90秒，allocated峰值1.92GB。全量阶段reserved峰值2.93GB。
五头×两诊断集共同特征评分29.66秒、allocated13.05MB，后者不含骨干/解码，
不能冒充标准CLI全链路显存或万级容量。

产物在`/var/tmp/deepfake-video-coverage-20261016/`；最小候选与新增原媒体单份
`raw-media.tar`另存`/mnt/deepfake-archive/C10-video-generator-coverage-20261016/`。
保留三个最佳头、实际配置/选择、完整分数、来源/排除和必要小适配器；旧媒体/骨干/缓存不重复归档。
骨干复用C6，前轮数据/适配器复用C9。index与SHA256SUMS逐文件登记原路径、归档路径、大小和SHA；
文件只读、团队可读，恢复需重映射原路径。归档保留原1500条库存（含未用于训练的排除组），
可重建受控媒体和原证据；不声称已完成跨实例恢复。

主43权重SHA：`916f57f547c83265528f033919f41f189161b207a20b57116fa8f6ce71e4f2fc`。
[control43](train-control43.json)、[expanded43](train-expanded43.json)、[expanded42](train-expanded42.json)
保留实际参数；训练复现使用归档适配器与固定源码，常规train入口不会自动设置本轮抽样预算。
[标准推理配置](infer-aigvd.json)使用持久权重与C6骨干，可通过现有infer入口加载：

```bash
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/video-generator-coverage-20261016/infer-aigvd.json
```

本轮完整结果为共同特征诊断；持久资产仅另做4条标准入口加载，不将它称为完整CLI验收。
下一阶段优先补更多真实来源/困难内容与保留旧域能力，进一步研究强表征的有限适配；
不继续在这两套已见dev上刷种子、帧数或融合系数，候选与许可固定后再衔接#54的最终验收。
