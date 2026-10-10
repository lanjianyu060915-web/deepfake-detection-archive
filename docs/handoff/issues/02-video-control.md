# [视频] 接入共享 ViF 数据并完成受控视觉基线

## 目标与依据

在尺寸/FPS/时长受控条件下取得冻结 ResNet18 媒体级头的双种子真实结果，比较视觉内容模型与简单元数据模型。历史 Hunyuan 受控 train618/dev146，真假各半；宽度EER50.00%、联合元数据53.42%，受控视觉模型尚未训练。

依据：[受控实验与配置](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/docs/experiments/vif-control-20261009/README.md)、[受控训练配置目录](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/docs/experiments/vif-control-20261009/)、[Git内的旧pilot清单](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/docs/experiments/vif-pilot-20261007/manifest.jsonl)、[固定视觉资源](https://github.com/lanjianyu060915-web/deepfake-detection-archive/blob/main/docs/research/pretrained-resources/README.md)。本任务自行完成视频输入接入，无需等待音频任务。

## 工作范围

- 复用 `/srv/shared-data/datasets/vif-panda/source_videos.zip`（约1.29GB、ZIP7040项）；完整SHA-256已与同目录 `provenance.json` 一致。在自己可写的任务目录按需安全解压，复用 `tools/prepare_vif_panda.py` 导入。来源当前为研究用途，许可限制随清单记录，不能写成已获正式比赛交付许可。
- 使用Git内 `docs/experiments/vif-pilot-20261007/manifest.jsonl` 恢复pilot归属（train716/dev154/holdout154）；这是登记元数据，不需读取holdout媒体或分数。复用 `tools/prepare_video_control.py`，先排除旧holdout及相连组件，再做train/dev内的宽高/FPS/1秒时长桶真假匹配；不读取holdout帧，不重新随机打乱旧归属。
- 接入固定 ResNet18 转换后的 `encoder.pt`，核对来源、许可和已登记哈希。优先恢复原资源，必要时按固定上游revision/转换方法取得；不在训练时隐式下载，不重复承担音频HuBERT工作。
- 复用 `train-hunyuan_geometry_duration-seed42.json` 和seed43配置：冻结ResNet18、4帧center_crop、轻量媒体级头。一次小profile后完成两个种子训练，保持对称预处理与相同dev。
- 重建元数据基线，报告实际样本数/组成、总体EER、连续分数、失败率、耗时/吞吐/显存。若未重现618/146，说明来源/匹配差异，不冒充原受控复现。

主要负责视频导入/受控准备工具及其测试、视频配置和新的实验卡。首次PR可以先完成资产导入与受控划分，再在本Issue交付训练结果；不要先引入大型视频框架。

## 验收

- [ ] ZIP与固定权重可复现接入，原来源/派生关系保留，旧holdout整组隔离、train/dev无已登记组跨分。
- [ ] 受控train/dev样本数、真假比例、来源/生成器分布和元数据基线有实际报告。
- [ ] 两个种子完成真实训练及全部受控dev评分，权重和连续分数可访问，逐样本失败有记录。
- [ ] 对同一受控dev比较视觉与元数据结果；不要求必须超过元数据模型，不将单生成器/小样本结果说成七生成器泛化。
- [ ] PR包含必要实现、配置、实验卡、相关行为检查及一轮标准检查。

部分微调/时序方法在冻结基线有实际结果后另开任务；本Issue不混入音轨融合，不触碰旧holdout验收。

## 环境与协作约定

当前服务器为 RTX 4090 24GiB，Python 3.12.15、torch 2.14.1+cu132；共享环境 `/opt/venvs/deepfake-cu132` 已可用，FFmpeg/ffprobe 已全局安装到 `/usr/local/bin`。复用现有依赖，不重新安装环境或修改驱动。

在自己的 clone/worktree 中，基于迁移修复分支 [`chore/cloud-migration-cleanup`](https://github.com/lanjianyu060915-web/deepfake-detection-archive/tree/chore/cloud-migration-cleanup) 建立 `issue/<本Issue编号>-<主题>`；该分支合并后改用最新 `main`。在仓库根目录使用统一入口：

```bash
scripts/devctl run python -m deepfake_detection_archive --help
# 代码变更先跑相关测试，PR前完成一次标准检查
scripts/devctl check
```

`/srv/shared-data/datasets` 为共享只读输入，目录/文件权限允许队友读取。派生媒体、清单、outputs/cache 使用自己可写的任务目录；需要把产物发布到 root 管理的共享目录时交由维护者处理，不因权限问题重下原包或放宽共享原媒体权限。大媒体、权重和完整分数不入 Git，PR 给出团队可读取的路径和必要哈希。

三条任务可独立进行数据准备和开发；只有一张 GPU，运行 GPU profile、特征提取或训练前在 Issue 记录预计占用时段，并与另一个模型任务错开。只修改本方向需要的代码；涉及公共训练器/编码器或依赖变更时先协调，避免同时修改。提交 LoRA 风格的小 PR，关联本 Issue，由负责人审查合并；不自行合并、扩容或正式参赛提交。

领取时在本 Issue 留下负责人、分支和第一步；未分派姓名，按方向领取。
