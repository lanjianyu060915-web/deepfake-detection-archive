# 当前项目状态

更新日期：2026-10-11。**音频 LogMel/HuBERT 双种子对照已完成；负责人已完成鲁棒性媒体准备及配对分数比较入口。尚无本届比赛成绩。**
本轮完成四次云端音频训练与完整dev评分；负责人通过 Issue 分工、审查 PR 并合并。
接手先读[个人交接](handoff/personal-cloud-codex.md)和[云端协作](handoff/cloud-collaboration.md)。

## 当前约定

- 音频、视频两项均参加；EER越低越好，内部`fake=1`、高分更伪造，保留逐样本连续分数。
- 用户已报名，官方没有额外信息，不再索取资料；未知项集中在[比赛要求](competition/requirements.md)。
  平台样例出现后按实际规范适配，不阻塞已授权研究。
- 优先有效训练、数据覆盖、EER和泛化；避免过度防御、重复校验和扩建证明体系。
- 已分配实例为 RTX4090 24GiB；在现有实例与权限内工作，不自动购买、扩容或正式提交。
  迁移修复及鲁棒性媒体准备/CI更新已通过PR #12、#13合并到`main`；音频、视频和数据三个Issue
  由队友推进，任务入口见[任务目录](handoff/issues/README.md)。Git遵循LoRA。

## 当前开发环境

- Python 3.12.15、uv 0.13.0、torch 2.14.1+cu132、torchvision 0.29.1+cu132；
  驱动595.91.07、CUDA Toolkit13.2，GPU为RTX4090 24GiB。
- `/opt/venvs/deepfake-cu132` 的124个环境依赖与锁文件匹配，dev/media/research齐全；
  `uv lock --check --offline`通过，锁定同步的只读预检显示无需改动。11项项目skills完整。
- 开发入口修复了继承其他checkout的`PYTHONPATH`导致源码串用的问题，`run/test/check`
  优先当前工作副本。共享环境同步显式排除本项目安装，CPU检查使用独立环境。
- 用户下载的BtbN FFmpeg包已核对SHA-256，版本为`N-127271-gba987fe24d-20261010`。
  按用户要求安装到root管理的`/opt/ffmpeg/btbn-628033703`，全局入口为
  `/usr/local/bin/ffmpeg`和`/usr/local/bin/ffprobe`，目录/程序权限允许所有账号读取和执行；
  用户级入口已移除，日常调用使用全局版本。
- 新环境探针复用项目实际FFmpeg/ffprobe解码路径；GPU前向、torchvision NMS、AMP、
  音频变换、确定性抽帧和研究工具合成检查通过。系统旧FFmpeg4.4.2会被该探针明确拒绝。
- 已恢复固定提交`9b33f5e`的ASVspoof2021只读评分参考；当前不需要恢复另外两套上游依赖。

音频分支合入最新main后的标准检查：**467项pytest通过（27.66秒，无跳过）**；另有3项视频工具测试，
Ruff、70文件格式检查、严格mypy（33+12文件）、Shell语法、sdist/wheel构建通过。
旧照片评测器7项stub不再默认执行，1,526个归档文件与恢复工具的字节完整性仍由pytest保护。
uv与构建后端统一到0.13系列，锁文件核对通过，运行时依赖未变。验证入口为`scripts/devctl check`。

CI已更新为独立CPU环境、固定并校验SHA-256的FFmpeg月底构建、合成环境探针和固定ASV评分参考，
与服务器共享GPU环境隔离。[PR #13的远端CI](https://github.com/lanjianyu060915-web/deepfake-detection-archive/actions/runs/38104568560)
已通过，实际使用Python3.12.15及CPU wheels。损坏音频及时间戳回归不再依赖不稳定fixture。
本轮18项配对回归、历史真实分数复算和独立评审通过；完整本地日志为
`outputs/score-comparison-20261011/project-check.log`，远端新增功能按对应PR的实际检查报告。

## 最新真实结果

| 工作 | 数据/协议 | 当前结果 |
| --- | --- | --- |
| 旧主机全量语音 LogMel42 | ASVspoof LA train25,379/dev24,844；batch256/BF16；30轮 | 历史dev EER 0.510204%，原checkpoint/分数未在当前云端恢复 |
| 云端 LogMel42/43 | 与旧实验相同清单SHA256；64,600采样点单片段 | dev EER **0.7849% / 0.9419%**；最佳第23/20轮，各24,844条评分0失败 |
| 云端冻结 HuBERT42/43 | 相同train/dev；BF16编码缓存、FP32分类头评分 | dev EER **1.4980% / 1.5306%**；最佳第26/20轮，各24,844条评分0失败 |
| 视频受控数据 | ViF Panda排除旧holdout154后probe4,458条，0失败 | Hunyuan受控train618/dev146，各自真假各半 |
| 视频元数据对照 | 上述Hunyuan受控dev | 宽度EER50.00%、联合元数据53.42%；视觉模型尚未训练 |
| 非语音辅助研究 | EnvSDD现有530条/106文件，106真424假 | 未训练；归属及非商用许可边界未完成，首批云端任务不依赖 |
| 压缩/重采样鲁棒性准备 | 4条ASV dev音频、4条ViF dev视频，各真假2/2；4种条件 | 16个派生文件转码及项目解码0失败，未进行模型评分 |
| 配对分数比较 | 历史ASV pilot 512条dev、四模型/种子已有评分 | 三组总体及18组生成器复算与历史一致，0失败；没有新推理 |

本轮冻结HuBERT未改善同域dev EER，保留LogMel42作为本轮较优候选；尚未进行新的鲁棒性或
未知生成器验证。四个checkpoint、连续分数及来源证据在团队可读目录
`/var/tmp/lanjianyu-issue9-20261011`；资源、分生成器EER和复现命令见
[云端双种子实验卡](experiments/asv-cloud-20261011/README.md)。

旧主机LogMel42训练器耗时1,519.13秒，包含准备和最终评分的完整实验1,614.62秒；
训练吞吐中位数863.85条/秒，CUDA allocated峰值3.24GiB、reserved5.37GiB，0跳过更新。
完整FLAC登记解码50,224条，0失败；跨划分高相关候选仅排除一条train音频，保留整个说话人。
结果、协议、代码版本和分生成器EER见[全量语音实验卡](experiments/asv-main-20261009/README.md)。

**以上EER均来自已知A01–A06生成器的语音开发集，不能代表比赛、通用音频或未知生成器泛化。**
旧主机run第30轮仍是最佳；本轮完成双种子观察，仍没有最终留出验收。
分数由最佳checkpoint对准备好的dev张量产生，耗时不等于端到端离线吞吐。

视频受控集主要由HunyuanVideo支持，不能称为七生成器受控评测。
尺寸/时长匹配降低了简单元数据捷径，但不证明其他偏差已消失。
旧154条holdout媒体及分数未触碰。详情见[视频受控实验卡](experiments/vif-control-20261009/README.md)。

负责人独立完成[鲁棒性媒体准备试跑](experiments/robustness-smoke-20261011/README.md)：
AAC、8 kHz带宽退化和H.264 CRF23/35，保留配对ID和失败分母，使用CPU。
队友候选交付后继续比较EER退化；媒体可解码不代表模型泛化有效。

配套[配对分数比较与历史复算](experiments/score-comparison-20261011/README.md)已完成：按ID对齐，
保留两侧全量失败分母，另报共同成功子集和ΔEER；可传同一冻结阈值诊断校准变化。
本轮音频候选已在上述任务目录交付，尚未进行新的媒体鲁棒性评分。

## 历史结果与工程

旧pilot仅作历史对照，划分与全量不同，不直接比较数字：

- 语音train486/dev512：LogMel42/43 EER21.48%/24.22%，AASIST-L25.78%/29.69%。
- 视频train716/dev154：冻结ResNet18 pad32.47%/32.47%，crop33.77%/31.17%；
  宽度基线24.75%，因此未把视觉结果解释为可靠生成伪迹。
- 65个CPU wheel及24条真实语音完成断网推理和推理恢复；这不是新全量模型或视频的离线验收。

入口：[方案](plans/implementation-plan.md)、[开发指南](development.md)、[论文与参考](research/literature.md)。
`uv.lock`固定Python3.12相关依赖；11项项目skills纳入Git；旧照片项目1,526个文件及恢复工具冻结。
当前已有EER、音视频解码、AMP训练、连续评分、推理恢复及打包。训练仍不支持optimizer/RNG续跑。
此前完成原生FLAC读取、跨音频SSL批量编码、平衡训练采样、最佳模型dev分数保存；
训练不再要求先独立预解码后又解码提特征，失败在准备阶段明确记账。

原电脑2026-10-10收尾检查：**400项pytest通过（15.70秒，无跳过）**；
3项视频工具测试、7项旧评测器stub、Ruff、58文件格式、严格mypy及构建通过。
检查入口为`tools/check_project.sh`。当时尚未运行远端CI，工程通过不等于模型泛化有效。

## 云端下一步与资产

1. [音频#9](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/9)：
   ASV/HuBERT接入及双种子对照完成，提交PR审查；后续安排候选泛化与离线路径验证。
2. [视频#10](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/10)：
   接入ViF/ResNet18资产，完成受控视觉双种子基线。
3. [数据#11](https://github.com/lanjianyu060915-web/deepfake-detection-archive/issues/11)：
   非语音及跨来源/生成器覆盖，可与模型任务并行。
4. [离线交付草稿](handoff/issues/04-offline-delivery.md)：尚未发布，候选冻结后启动。

三个已发布任务由队友推进，均明确独立职责、实际输入与验收，不要求重新安装共享环境。
单GPU使用在Issue中协调，负责人审查PR后合并。

2026-10-10核对时，本工作副本及服务器主checkout尚未接入资产清单中的7个训练路径，
包括HuBERT/ResNet18权重、旧LogMel42 checkpoint和固定实验清单。队友后续接入进度以各Issue为准。
当时共享盘`/srv/shared-data`已有：

- ASVspoof2019 LA Parquet：train25,380/dev24,844行，元数据可读，未物化原媒体；
  来源记录要求核对官方protocol，且train比旧训练多一行，不能直接视为同划分复现。
- ViF Panda原始ZIP约1.29GB，目录可读、7,040项，未解压和导入，许可仍限定研究用途。
- FakeSound ZIP约986MB，未解压；仅fake诊断集，不能单独训练二分类或选择阈值。

两个ASV Parquet及ViF ZIP的SHA-256与已有来源登记一致；FakeSound来源登记没有参考哈希，
只检查了归档目录。共享数据为root:dlteam，文件0644、目录2755，路径及ACL允许队友读取；
原媒体保持共享只读，实验产物写入各自工作副本。
上述核对没有重验全部媒体、标签或许可。
上述历史核对时另有官方原包下载进程，本轮未重启或复制下载。
2026-10-11音频任务已接入媒体、清单和HuBERT，四次真实训练完成；视频资产仍按其Issue接入。

[资产目录](handoff/local-assets.json)保留原电脑快照及状态包哈希，并补充当前路径核对。
历史Git bundle和约4.92MiB状态包可作为恢复来源；当前工作副本没有这些包，上传状态按实际
核对。缓存默认重建，归档、历史记录和原始资产保持不变。
原收尾记录见[工作记录](worklog/2026-10-10-cloud-handoff.md)。
