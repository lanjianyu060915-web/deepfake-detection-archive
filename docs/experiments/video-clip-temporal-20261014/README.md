# 冻结CLIP特征的轻量时序头

复用已有Conv1d时序头训练两种子；原dev选定seed42后，新来源受控dev
EER31.17%，原MLP43为31.33%，仅改善0.16个百分点；去Open-Sora后
29.85%→30.29%退化。**保留MLP43，不采用本轮时序头替换候选**，停止轻头/帧数调参，
下一投入为多来源训练和困难内容覆盖。

## 方法与选择

沿原train3614/dev844，4帧CLIP ViT-L/14表征、同一预处理和媒体。
`media_feature_head_v1`的`temporal_kernel_size=3`启用项目已有TemporalHead：
LayerNorm→kernel3 Conv1d→GELU/dropout→1×1 Conv1d，帧logit均值直接输出。
默认0仍为旧MLP，旧checkpoint与ResNet时序头参数key保持兼容。

骨干冻结，隐藏64、dropout0.1、batch128、BF16、AdamW lr1e-3/wd1e-4，
max30/patience6，OMP/MKL=2。各seed仅按原844条dev早停，再按较低EER选择，平局42；
新来源dev不用于epoch/seed选择，保留部分未读取。时序头149121参数，容量同时增长，
本轮比较完整候选，不能将差异单独归因于时间关系或运动。
输入仅为4帧全局图像特征，未显式编码实际时间间隔，时序覆盖仍有限。

| 种子 | 原dev最佳EER | 最佳epoch | 实际epoch | head训练秒数 |
| --- | ---: | ---: | ---: | ---: |
| 42（选定） | 23.9336% | 14 | 20 | 41.64 |
| 43 | 24.4076% | 6 | 12 | 36.03 |

两轮均early stopping、exit0；新来源评分前已冻结选择。

## 显式消费旧预计算特征

特征来自原`865a2c9`，4455个唯一缓存对覆盖4458条train/dev记录。
当前入口后来增加`stream_index:null`，使缓存配置/整文件签名变化；
未改写旧identity或假称由新提取器生成，也未为此重提主干。

在缓存进入训练前完成一次完整验证；之后任务私有`precomputed_train.py`直接调用现有
`train_model`、`_CachedDataset`、`_write_dev_scores`，epoch内按原路径读取。
任务独立缓存副本在实验完成后已清理；原只读缓存仍在原CLIP任务目录。
训练源码为`3a4c0c8`，适配器SHA及旧pipeline来源见[核心摘要](summary.json)。
配置保存真实输入参数；本次实际入口是归档适配器，不能把常规`train`的当前缓存身份
当作本轮旧特征来源。没有新增公共训练框架或依赖。

## 完整受控比较

与[新来源四候选实验](../video-aigvdbench-20261014/README.md)复用同一1880条dev，
统一448×256/16fps/1.25秒/CRF23。同骨干、4帧、224 native center crop、
FP32、batch4、mean logits；旧MLP43完整分数直接复用。时序头使用原dev选定的42。
两模型均1880/1880、零失败。

| 条件 | MLP43 | 时序42 |
| --- | ---: | ---: |
| pooled | 31.3333% | 31.1719% |
| 去Open-Sora | **29.8529%** | 30.2941% |
| 600组Open-Sora/real配对 | 32.6667% | 32.3333% |

每个生成器与同一600条真实池比较；Open-Sora600假，其余各40假，
Opensora40单列。小桶仅描述，各行共享真实池，不相加或当独立重复实验。

| 生成器 | MLP43 | 时序42 |
| --- | ---: | ---: |
| AccVideo | 25.67% | 25.00% |
| AnimateDiff | 22.50% | 20.00% |
| EasyAnimate | 32.50% | 30.00% |
| Gen2 | 32.50% | 32.67% |
| Gen3 | 35.00% | 35.00% |
| IPOC | 20.67% | 22.50% |
| Jimeng | 30.00% | 28.67% |
| Luma | 42.50% | 37.17% |
| Open-Sora | 32.67% | 32.33% |
| Opensora | 33.67% | 30.00% |
| Pyramid-Flow | 25.00% | 25.00% |
| RepVideo | 32.50% | 33.50% |
| SEINE | 42.50% | 40.83% |
| SVD | 35.00% | 35.00% |
| Sora | 31.33% | 26.50% |
| VideoCrafter | 22.17% | 22.50% |
| causvid_24fps | 26.83% | 23.17% |
| vidu | 35.00% | 36.50% |

部分生成器改善，但逐桶各自阈值的EER不能代替整体共同排序。
整体收益小、去Open-Sora退化，尚不足以采用；不据原dev改善继续扫头或重选种子。

## 资源与恢复

两步batch128 BF16 head-only profile通过；训练峰值allocated9.20MB、reserved27.26MB，
不含冻结主干。完整推理端到端408.37秒、runner389.95秒，进程最大RSS3386420KiB；
本次没有收集准确推理CUDA峰值，不用head峰值冒充整个模型。OMP/MKL=2且共享负载
不同，不将它与旧726.69秒的差异归因时序头，不把1880条投影当万级容量验收。

实际配置、权重、完整分数与日志在`/var/tmp/deepfake-video-clip-temporal-20261014/`，
本人输出已设dlteam可读。最小24文件、6.92MB集合（双seed最佳权重、选择/历史、适配器、
配置与分数）另存`/mnt/deepfake-archive/C7-video-clip-temporal-20261014/`，
骨干复用C6；不备份重复主干、原媒体或缓存。选定权重SHA256：
`45bb5050bc7058f8be102669a153d028484bcda192429dc893bb552c0f737265`。

标准推理使用[实际配置](infer-controlled.json)：

```bash
scripts/devctl run python -m deepfake_detection_archive infer \
  --config docs/experiments/video-clip-temporal-20261014/infer-controlled.json
```

恢复时按C7索引映射权重/配置及C6骨干；重跑使用新输出目录。训练复现使用C7保存的
适配器、[seed42](train-seed42.json)/[seed43](train-seed43.json)配置与固定训练源码，
所需旧缓存可从既有副本取得，缺失时按原提取源码重建；路径变更登记新的适配器SHA。

新来源与旧Panda训练/选择集的YouTube母媒体交集未知，底层许可未确认。
本轮没有重跑自然组、历史诊断或读取holdout，研究结论不作为比赛成绩或发布晋级依据。
