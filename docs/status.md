# 当前项目状态

更新日期：2026-10-10。**全量语音基线已完成，后续开发和训练转到云端。尚无本届比赛成绩。**
负责人通过 Issue 分工、审查 PR 并合并。本地本轮收尾，不再启动新训练。
接手先读[个人交接](handoff/personal-cloud-codex.md)和[云端协作](handoff/cloud-collaboration.md)。

## 当前约定

- 音频、视频两项均参加；EER越低越好，内部`fake=1`、高分更伪造，保留逐样本连续分数。
- 用户已报名，官方没有额外信息，不再索取资料；未知项集中在[比赛要求](competition/requirements.md)。
  平台样例出现后按实际规范适配，不阻塞已授权研究。
- 优先有效训练、数据覆盖、EER和泛化；避免过度防御、重复校验和扩建证明体系。
- 云平台、规格和预算尚未确定；在已分派实例与权限内工作，不自动购买、扩容或正式提交。
  当前本地收尾未推送、发布Issue或创建PR。Git遵循LoRA。

## 最新真实结果

| 工作 | 数据/协议 | 当前结果 |
| --- | --- | --- |
| 全量语音 LogMel42 | ASVspoof LA train25,379/dev24,844；batch256/BF16；30轮 | **dev EER 0.510204%**，最佳第30轮；24,844条评分0失败 |
| 全量语音 LogMel43、HuBERT42/43 | 相同清单与单片段64,600采样点；实值配置已保存 | 尚未训练，转云端Issue01 |
| 视频受控数据 | ViF Panda排除旧holdout154后probe4,458条，0失败 | Hunyuan受控train618/dev146，各自真假各半 |
| 视频元数据对照 | 上述Hunyuan受控dev | 宽度EER50.00%、联合元数据53.42%；视觉模型尚未训练 |
| 非语音辅助研究 | EnvSDD现有530条/106文件，106真424假 | 未训练；归属及非商用许可边界未完成，首批云端任务不依赖 |

LogMel训练器耗时1,519.13秒，包含准备和最终评分的完整实验1,614.62秒；
训练吞吐中位数863.85条/秒，CUDA allocated峰值3.24GiB、reserved5.37GiB，0跳过更新。
完整FLAC登记解码50,224条，0失败；跨划分高相关候选仅排除一条train音频，保留整个说话人。
结果、协议、代码版本和分生成器EER见[全量语音实验卡](experiments/asv-main-20261009/README.md)。

**0.51%只来自已知A01–A06生成器的语音开发集，不能代表比赛、通用音频或未知生成器泛化。**
第30轮仍是最佳轮次，未触发早停；没有最终留出验收或两种子稳定性结论。
分数由最佳checkpoint对准备好的dev张量产生，耗时不等于端到端离线吞吐。

视频受控集主要由HunyuanVideo支持，不能称为七生成器受控评测。
尺寸/时长匹配降低了简单元数据捷径，但不证明其他偏差已消失。
旧154条holdout媒体及分数未触碰。详情见[视频受控实验卡](experiments/vif-control-20261009/README.md)。

## 历史结果与工程

旧pilot仅作历史对照，划分与全量不同，不直接比较数字：

- 语音train486/dev512：LogMel42/43 EER21.48%/24.22%，AASIST-L25.78%/29.69%。
- 视频train716/dev154：冻结ResNet18 pad32.47%/32.47%，crop33.77%/31.17%；
  宽度基线24.75%，因此未把视觉结果解释为可靠生成伪迹。
- 65个CPU wheel及24条真实语音完成断网推理和推理恢复；这不是新全量模型或视频的离线验收。

入口：[方案](plans/implementation-plan.md)、[开发指南](development.md)、[论文与参考](research/literature.md)。
`uv.lock`固定Python3.12相关依赖；11项项目skills纳入Git；旧照片项目1,526个文件及恢复工具冻结。
当前已有EER、音视频解码、AMP训练、连续评分、推理恢复及打包。训练仍不支持optimizer/RNG续跑。
本轮完成原生FLAC读取、跨音频SSL批量编码、平衡训练采样、最佳模型dev分数保存；
训练不再要求先独立预解码后又解码提特征，失败在准备阶段明确记账。

2026-10-10收尾统一检查：**400项pytest通过（15.70秒，无跳过）**；
3项视频工具测试、7项旧评测器stub、Ruff、58文件格式、严格mypy及构建通过。
检查入口为`tools/check_project.sh`。远端CI尚未运行，工程通过不等于模型泛化有效。

## 云端下一步与资产

1. [Issue00](handoff/issues/00-cloud-setup.md)：环境、持久卷和Git外资产；代码clone不会带来数据或权重。
2. [Issue01](handoff/issues/01-audio-comparison.md)：LogMel43和HuBERT双种子同划分对照。
3. [Issue02](handoff/issues/02-video-control.md)：受控视觉基线，再按收益做最小微调/时序对照。
4. [Issue03](handoff/issues/03-generalization-data.md)：非语音及跨生成器/来源覆盖。
5. [Issue04](handoff/issues/04-offline-delivery.md)：候选冻结后做双分支代表性离线验收。

代码另有完整Git bundle，可在服务器恢复当前分支；迁移状态包约4.92MiB，
包含必要清单、LogMel42权重/分数及记录；原媒体和预训练权重另传。
[资产目录](handoff/local-assets.json)给出实际路径与包哈希。**均尚未上传到云端。**
缓存默认重建，现有数据和历史记录原地保留，没有删除或批量搬动归档。
收尾记录见[工作记录](worklog/2026-10-10-cloud-handoff.md)。
