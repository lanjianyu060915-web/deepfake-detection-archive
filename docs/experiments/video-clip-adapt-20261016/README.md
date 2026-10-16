# CLIP 最后 block 的有限适配

有限微调的固定主43在AIGVD已见受控dev为 **17.8333%**，C10为18.8281%、同预算头对照18.5156%；
去五个训练标签为19.3333%（C10 21.00%）。旧Panda dev24.8815%→23.2227%。
但ViF pooled25.9079%→25.9615%基本持平，复现42的ViF pooled26.9231%、
未训练标签30.3601%均退化。**保留tuned43作研究候选，C10仍为通用视频主候选**；
不据外部结果改选种子，不升级发布默认，停止本轮已见dev参数扫描。
完整计数及逐生成器结果见[摘要](summary.json)。

## 方法与选择

承接[三生成器覆盖扩展](../video-generator-coverage-20261016/README.md)，
复用已清理的train5412/dev1344，不新增媒体、不改变划分或读取holdout。
旧Panda train3614/dev844，新AIGVD train1798/dev500；已确认跨split重复原组的
两条训练记录仍被排除。未知Panda原YouTube关系保持未知，已见外部dev只作诊断。

骨干为同一C6本地CLIP ViT-L/14，头初始化为C10 expanded43的H64/dropout0.1 MLP。
固定前23个视觉Transformer block及visual projection，只训练最后一个block、
末端post-LayerNorm和分类头。冻结projection的权重，保留经它传回最后block的梯度。
不做L2归一化，视频仍按4帧logit均值输出连续分数。
采用已安装的[Transformers固定实现](https://github.com/huggingface/transformers/blob/c4c4605f961877852b89b97478536bb3a26b6fd1/src/transformers/models/clip/modeling_clip.py)：
前缀包含embedding、pre-LayerNorm及前23层，保留CLS和全部patch tokens，
后缀执行最后层、CLS post-LayerNorm、projection和分类头。

真假使用同一4帧224 native center crop。缓存为FP32 `[4,257,1024]`，不做量化；
训练前只准备并验证一次，不在epoch内重复哈希或运行前23层。
模型支持原视频张量及缓存tokens两种路径，完整checkpoint通过现有infer/predict加载；
不引入新依赖或另一套训练框架。

| 组 | 最后block及post-LN | 初始化 | 用途 |
| --- | --- | --- | --- |
| baseline-C10 | 冻结 | C6骨干+C10头 | 原冻结研究候选，无额外训练 |
| control43 | 冻结 | 相同 | 匹配继续训练头的预算 |
| tuned43 | 训练 | 相同 | 预先固定的主候选 |
| tuned42 | 训练 | 相同 | 第二种子复现，不据外部结果改选 |

三个训练均来源×label等权有放回抽样，每epoch5412次、batch128、保留尾batch，
8完整epoch、344次optimizer更新、43296次抽样、patience8。
AdamW头lr1e-4、骨干lr1e-5、weight decay1e-4、BF16 AMP、grad norm1、TF32关闭。
epoch仅由共同开发集旧844/新500的source macro EER选择。
primary43在诊断前固定；两个种子及control全部冻结后才打开外部诊断。

固定源码为`b2ef85761930fc5aa84fa89785078ae0a36a423d`，运行副本
`/var/tmp/deepfake-video-clip-adapt-src`；实际入口为本轮私有train.py和diagnose.py，
复用现有train_model、缓存DataLoader和评分器。实验配置记录真实模型及TrainConfig，
不把任务适配器配置冒充公共train CLI配置。

## 冻结诊断与取舍

所有四个模型各覆盖共同dev1344、AIGVD1880、ViF1949，零失败。外部集合均已见，
以下为描述性诊断，不是盲测或比赛成绩；不同视图共享真实池，不能相加为独立重复实验。
EER单位为百分比，低更好。

| 视图 | real/fake | C10 | control43 | 主tuned43 | 复现tuned42 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧Panda dev844 | 422/422 | 24.8815% | 24.8815% | 23.2227% | 23.9336% |
| 新域dev500 | 100/400 | 18.0000% | 18.0000% | 18.0000% | 17.2500% |
| 原新域dev200 | 100/100 | 18.0000% | 18.0000% | 18.0000% | 18.0000% |
| 新增三生成器dev400 | 100/300 | 18.0000% | 18.0000% | 18.0000% | 17.0000% |
| 共同dev1344 | 522/822 | 22.5061% | 23.3716% | 21.4559% | 21.6475% |
| AIGVD1880 | 600/1280 | 18.8281% | 18.5156% | 17.8333% | 18.7500% |
| AIGVD去Open-Sora1280 | 600/680 | 21.3333% | 20.8333% | 19.6667% | 21.1667% |
| AIGVD去五训练标签1120 | 600/520 | 21.0000% | 20.7692% | 19.3333% | 20.7692% |
| ViF1949 | 104/1845 | 25.9079% | 26.3957% | 25.9615% | 26.9231% |
| ViF未训练原标签1326 | 104/1222 | 29.2144% | 29.1326% | 28.6416% | 30.3601% |

AIGVD去五标签排除SEINE、Open-Sora、AccVideo、EasyAnimate、SVD，剩13个生成标签；
Open-Sora/Opensora等家族可能重叠。ViF视图沿用原Panda训练标签的排除规则，
不是已确认独立生成器家族或独立母媒体。ViF只有104条real对1845条fake，
小差异不能支持稳定排名。

主43相对C10的ViF未训练标签改善约0.57点，仍差于最早冻结CLIP的27.8846%；
rep42在此视图30.3601%反而退化。不能把AIGVD改善称为全面泛化改善。
原新域dev200保持18%，未恢复前轮#77的16%。control的来源macro与C10相同，
pooled却变差，说明只看共同来源macro会遗漏跨来源分数对齐差异。

AIGVD逐生成器共享600条real（Open-Sora600fake，其余各40fake）。主43在AccVideo
15.50%→12.50%、Luma28.83%→27.50%、VideoCrafter15.00%→11.67%；
RepVideo18.50%→19.33%、SVD17.50%→18.17%、Sora24.17%→25.00%退化。
保留全部生成器桶于摘要，不只报告改善项。

| 训练 | 所选epoch | 共同source macro EER | 训练阶段秒 |
| --- | ---: | ---: | ---: |
| control43 | 3 | 21.4408% | 377.94 |
| tuned42 | 2 | 20.5918% | 386.24 |
| tuned43 | 4 | 20.6114% | 400.16 |

三个训练实际预算均为8epoch/344updates/43296draws，零跳过更新。
小样本标准infer使用主43完整checkpoint，4/4、零失败，raw与本轮cached dev logits最大差
4.65e-6；这只是加载与数值对齐，不代表完整CLI容量验收。完整外部结果来自共同前缀缓存评分。

## 资源与恢复

训练前用128条train-only真实视频profile batch32/64/128，分别约717/744/761段每秒，
选择batch128、无需梯度累积；BF16训练allocated峰值13.78GB、reserved19.65GB。
8条实际视频FP32 raw/cached logits最大差0，profile模型不作为训练初始化。
profile只测后缀训练吞吐，不代表原视频端到端推理速度。

产物位于`/var/tmp/deepfake-video-clip-adapt-20261016/`。
缓存只用于本轮运行，诊断每个视图提一次、四模型共用、完成后删除该临时诊断缓存；
不重复归档媒体或约28GB token缓存。原数据、来源及排除记录复用C9/C10，初始骨干复用C6。
许可范围继续为内部研究，底层媒体及模型正式赛用许可尚未确认。

全量前缀准备428.65秒、28,451,327,667字节、零失败；训练阶段总计1702.10秒
（含准备、基线评分、三训练及初始化），诊断总计428.78秒。
两个外部视图分别207.36/218.18秒，含prefix准备136.54/146.49秒；
诊断allocated峰值11.48GB、reserved13.65GB。训练最大allocated13.78GB、reserved19.66GB。
这些进程级数字不代表正式离线包万级端到端容量。

最小候选另存 `/mnt/deepfake-archive/C11-video-clip-adapt-20261016/`：
三个自包含最佳checkpoint、实际配置/历史、完整分数、来源清单、摘要及必要适配器；
文件只读、团队可读，index/SHA256SUMS登记原路径、大小和哈希。
原媒体及初始权重复用C6/C9/C10；全部评分及归档完成后，28.45GB本轮训练tokens已释放，
需要时按固定适配器重建。跨实例恢复尚未演练。

主43权重SHA：`a67f615f1e1e2f45bf059cc19b2962edc3df5b7085f7f82505ab96cb35361cbf`。
实际训练配置：[control43](train-control43.json)、[tuned43](train-tuned43.json)、[tuned42](train-tuned42.json)。
完整模型无需额外encoder，[推理配置](infer-aigvd.json)可经现有入口运行：

```bash
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/video-clip-adapt-20261016/infer-aigvd.json
```

下一步优先更多真实来源和困难内容，与#51数据线衔接；主43/C10均保留作新数据对照，
不继续在这两套已见dev上扫epoch、种子或融合系数，不重新使用保留集作开发集。
