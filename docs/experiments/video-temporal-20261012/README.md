# 视频轻量时序头

在原 Hunyuan train618/dev146 上完成两个种子的实际训练，保留 ResNet18 `layer4`
微调和四帧输入，只将逐帧 MLP 换成邻帧卷积头。**汇总跨来源分数小幅改善，
但同域双种子没有稳定优势，生成器间存在退化；保留研究候选，不替换 v0.1.0。**

## 方法与选择

- 沿用原清单、来源和初始化权重；4帧、224×224 `center_crop`、batch8、BF16 AMP，
  AdamW 主干 `1e-4` / 头 `1e-3`，最多20 epoch，patience5，无新增数据或增强。
- `temporal_kernel_size=3`：帧特征恢复为 `[B,T,512]`，依次经过 LayerNorm、
  Conv1d(kernel3)、GELU、Dropout、Conv1d(kernel1)，再对帧 logit 取均值。
  只训练 `backbone.layer4.*` 和 `head.*`，更早主干保持冻结。
- 每个种子只按 dev EER 选 epoch；再按最低 dev EER 选候选，相同时选 seed42。
  因而先冻结 seed43/epoch8，再评分原固定310条尺寸匹配诊断及其 CRF35 视图。
  没有读取 holdout，也没有依据跨来源分数重新选种子或参数。

## 结果

| 结构 | seed42 dev EER | seed43 dev EER | 最佳 epoch（42 / 43） |
| --- | ---: | ---: | ---: |
| 原逐帧头 | 16.4384% | 16.4384% | 3 / 3 |
| 时序头 | 17.8082% | 16.4384% | 3 / 8 |

两轮均为618条训练、146条dev，dev覆盖率1.0、失败0；分别在第8、13 epoch早停。

| 固定尺寸匹配诊断（310条，假221 / 真89） | 原发布候选 seed42 | 冻结时序候选 seed43 | ΔEER（百分点） |
| --- | ---: | ---: | ---: |
| 原始媒体 | 30.7692% | 29.2135% | −1.5557 |
| CRF35，448×256 | 32.5843% | 30.3371% | −2.2472 |

两视图均310/310成功、0失败，同一组 sample_id 的配对覆盖率1.0。
这里比较的是已选择候选与现有发布候选，不是相同种子的结构因果对照。

生成器分组按原221个匹配对计入各自真实父片段，真实片段可跨组复用：

| 生成器 | 对数 | 原始：原头 → 时序头 | CRF35：原头 → 时序头 |
| --- | ---: | ---: | ---: |
| pixverse-v4-5 | 62 | 20.968% → 25.806% | 25.806% → 25.806% |
| hailuo | 55 | 25.455% → 23.636% | 29.091% → 25.455% |
| kling-v1 | 50 | 32.000% → 32.000% | 30.000% → 32.000% |
| gen4-turbo | 38 | 47.368% → 47.368% | 44.737% → 47.368% |

Wan2.1-VACE（11对）和 Wan2.2-I2V（5对）两候选分组 EER 均未变，样本太少，不作有效性结论。
汇总 EER 改善没有在各生成器一致出现，不能据此声称普遍泛化更好。

## 资源与产物

- RTX4090；实际缓存帧短测 batch8 / BF16，预热后3步平均9.60ms，峰值 allocated
  238.49MiB、reserved310MiB。短测使用平衡占位标签，只确定资源，不计为模型结果。
- seed42总206.8s（含首次解码建缓存），seed43总48.6s（复用764份缓存）；
  每轮含dev约3.0–4.4s。第二种子最初遇到已有 tuple/JSON-list 身份比较问题，
  修复后在新目录重启；张量内容和缓存 key 未变，没有重复解码。
- 产物在团队可读 `/var/tmp/deepfake-video-temporal-20261012/`：
  `seed42/`、`seed43-retry/` 含最佳权重、dev逐样本分数和训练摘要；
  `scores-matched-{original,crf35}/` 含冻结候选分数；`compare-*.json`、
  `per-generator.json`、`summary.json` 为比较和汇总。帧缓存完成两轮后清除，可按配置重建。
- 冻结 seed43 权重：`seed43-retry/training/best.pt`，SHA256
  `977721c2edd496c6e790ea56df41a34b1402399ebd8f414f754b0dfd4ea1cef1`。
  数据、encoder与许可沿用[原微调实验](../video-finetune-20261011/README.md)及其来源登记，
  仍限已登记的本地非商业研究用途，没有取得正式比赛使用结论。

训练用本目录 `seed42.json`、`seed43.json`，冻结后评分用 `scores-matched-*.json`：

```bash
scripts/devctl run python -m deepfake_detection_archive train --config <训练配置>
scripts/devctl run python -m deepfake_detection_archive infer --config <评分配置>
```

输出目录必须为空；复现实验应换新输出目录。固定诊断清单、CRF35媒体及原候选分数已从
原实验工作副本原样复制至上述根目录的 `diagnostic/`，没有重新筛选配对。

## 限制与下一步

头参数33,921→99,457，容量和邻帧建模同时变化；四帧为全片稀疏采样，当前结果不能归因为
学到了运动伪造特征。dev仅146条且已用于多轮选择，固定诊断仅6生成器、仍有时长混杂，
不是独立最终验收；具体限制见[尺寸控制实验](../video-size-control-20261012/README.md)。

新头须使用本次代码及冻结的四帧/224预处理配置，沿用 `video_frames` 本地推理，
无需训练期 encoder；v0.1.0旧运行包不能读取新字段。默认 kernel1 保留旧模型与参数键。
下一投入是 #40 多生成器数据训练，再比较是否保留时序头；不围绕当前小dev反复调参。
