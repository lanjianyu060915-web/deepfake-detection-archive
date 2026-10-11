# ViF 云端受控视觉双种子对照

2026-10-11，RTX4090，Issue #10 / PR #18。
冻结ResNet18帧头在受控dev上的EER为20.55%/21.92%，优于同集元数据对照。

## 方法与结果

ViF Panda的HunyuanVideo子集按宽、高、FPS和整秒时长桶匹配真假，
保留旧pilot归属；train618（real309/fake309）、dev146（real73/fake73）。
导入时复制原媒体字节，旧154条holdout在probe前整组排除，未用于帧解码、选模或评分。
保守文件名关系组不等于确认身份；匹配方法见[受控数据准备](../vif-control-20261009/README.md)。

固定ResNet18-a1，4帧224×224 center_crop，FP32冻结特征；隐藏层64、dropout0.1，
帧logit均值得到媒体logit。头训练batch64、BF16 AMP、AdamW lr0.001、
weight_decay0.0001，最多20轮、patience5；dev选最佳轮次后FP32评分。

| 模型 | 最佳/完成轮次 | dev EER | 相对联合元数据ΔEER | 成功/失败 |
| --- | --- | ---: | ---: | ---: |
| 宽度元数据 | — | 50.000000% | — | 146/0 |
| 联合元数据 | — | 53.424658% | — | 146/0 |
| ResNet18帧头42 | 4/9，早停 | 20.547945% | −32.876712个百分点 | 146/0 |
| ResNet18帧头43 | 4/9，早停 | 21.917808% | −31.506849个百分点 | 146/0 |

训练无跳过更新，特征准备无失败。完整实验42为179.95秒，包含764条首次特征缓存；
43复用缓存为6.81秒。训练器耗时、allocated显存峰值及必要资产哈希见[结果摘要](summary.json)，
不能把缓存后头训练或评分速度解释为端到端离线吞吐。

## 资产与复现

固定来源、权重和许可见[数据说明](../../data.md)与[资源准备工具](../../research/pretrained-resources/README.md)。
团队资产根目录 `/var/tmp/lanjianyu-issue10-20261011`：

| 相对资产根目录 | 内容 |
| --- | --- |
| `vif-panda-research/`、`control/` | 原媒体、来源和受控清单/元数据分数 |
| `resnet18-a1/` | 本地编码器及许可 |
| `seed42/`、`seed43/` | `training/best.pt`、连续dev分数、训练历史和日志 |

[seed42](seed42.json) / [seed43](seed43.json) 配置中的
`data/vif-panda-research`、`models/pretrained/resnet18-a1` 分别指向对应资产；
manifest位于 `outputs/experiments/vif-cloud-20261011/control/`。
可挂载上述资产或在配置副本中填写实际路径，`output_dir`、`feature_cache` 使用自己的新任务目录，
再调用项目 `train --config`。源码版本、清单/来源及checkpoint/分数标识集中在摘要，
路径相对资产根目录；无需重建数据即可使用已交付候选。

需要重新准备受控集时，从固定历史恢复pilot，传给
`tools/prepare_video_control.py --pilot "$assets/pilot-manifest.jsonl"`：

```bash
assets=/var/tmp/your-new-video-task
mkdir -p "$assets"
git show 4bbd292:docs/experiments/vif-pilot-20261007/manifest.jsonl \
  > "$assets/pilot-manifest.jsonl"
```

## 结论与下一步

结果仅限同来源、Hunyuan单生成器小dev，指定元数据匹配不能证明其他来源/编解码捷径已消除。
dev用于早停，未作最终holdout验收；Panda/HD-VILA及派生权重限本地非商业研究，正式赛用尚未确认。
先做压缩与独立来源/生成器评分，再根据误差决定部分微调或轻量时序。
