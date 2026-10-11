# 非语音数据接入与泛化协议（Issue #11）

日期：2026-10-11；范围：数据准备、来源核查、划分和训练输入验证。
本轮没有训练检测模型、评分最终 holdout 或产生比赛成绩。

## 已交付内容

- FakeSound 固定原包的诊断清单：实际媒体、缺失原始录音、逐文件哈希和全量解码结果。
- Doppelganger/FSD50K 小规模环境音配对子集：真实来源与生成版本采用同一处理，
  具有来源登记、逐样本归属、分组 train/dev/holdout，可由现有训练器读取。
- 来源/生成器留出设计及视频补充来源入口；单生成器 pilot 不构成泛化证明。

实际数量、耗时和资产哈希见本目录 `results.json`。所有媒体、完整清单、归属、审计及原始
许可快照在团队可读的 `/var/tmp/lanjianyu-issue11-20261011/`；这是当前实例的任务目录，
需要纳入平台持久化/迁移时，由维护者复制到其管理的共享持久卷。本任务未改变共享原包权限。

最终配对子集如下，各行真假数量相等；共有 153 个连通组。

| 划分 | real | fake | 连通组 |
| --- | ---: | ---: | ---: |
| train | 161 | 161 | 105 |
| dev | 35 | 35 | 24 |
| holdout | 35 | 35 | 24 |

实际下载 572 条原始音频、457,121,938 bytes，全部成功解码；55 条真实录音不足 5 秒，
连同其生成版本排除，最终 462 条可用（80.77%）。初次下载/处理 324.54 秒，修正窗口后
复用已验证下载重处理 3.28 秒，没有再次下载。`doppelganger-v2` 是最终输入；`v1` 仅保留
下载及中间记录。八个内容桶均出现在各 split，但 dev 的 WIND 只有 1 对，不作稳定分组结论。
392 条 train/dev 已通过现有训练解码/切段接口读取，0 失败；此检查没有读取 holdout 波形。
全部输出是 16 kHz、单声道、64,600 点、PCM16、129,244 bytes；dev 文件大小这一常量
元数据基线 EER 为 50%，不代表检测模型 EER，也不证明其他捷径消失。

最终 `scripts/devctl check`：425 项 pytest 通过、1 项因缺少固定 ASVspoof 参考 checkout
跳过；Ruff、严格 mypy、3 项视频工具测试、7 项历史 stub 和构建通过。首次检查中既有
ASV 归档 `touch()` 时间戳测试失败，单独复查与最终整轮均通过；本任务没有修改该测试或
ASV 导入代码，保留两次日志供追溯。

## FakeSound：1,899 个实际伪造片段，原始媒体未提供

输入为 `/srv/shared-data/datasets/fakesound/FakeSound.zip`，986,570,159 bytes，SHA-256
`96cfc28b1e2274ff3e4404be94c4039645e862376ec161549386a9e0577c0623`。
ZIP 共 1,909 项，其中 1,899 个 WAV、4 个 JSON，其余为目录。

元数据共有 3,798 行，包含 1,899 个带重建区间的伪造项及 1,899 个原始项；只有前者存在。
全部实际 WAV 完整解码成功，解码失败 0；以所有元数据行为分母的媒体/解码覆盖率为 50%。
135 个 WAV 为 16 kHz，1,764 个为 32 kHz，均为单声道 PCM16。工具保留原字节，不做转码。

有两个不能忽略的上游差异：

1. JSON 的 `FakeSound/deepfake_data/` 与 ZIP 的 `FakeSound/audio_data/` 前缀不同。
   导入只执行这个明确映射，不把错误路径当作全部媒体缺失。
2. [官方 README](https://github.com/FakeSoundData/FakeSound)称 raw `label=0` 为 deepfake，
   但这个固定 ZIP 所有已发布 manipulated WAV 对应 raw `label=1`、非零重建区间；
   raw `label=0` 均为缺失的原始项。内部映射依据官方“只发布 manipulated”说明、重建区间
   与显式父项对应关系；完整 raw 标签及冲突留在 provenance/summary，不套用 README 的
   数字方向。HF 自动生成的四个文件夹类别同样不是真假标签。

按 `audio_id` 去除明确的 `onset_offset` 后关联原始项，共 1,718 个组，其中 135 个跨
上游 difficulty 桶。因此这些桶不能直接充当独立划分。所有记录保持 `unassigned`。
生成器、身份及逐条内容类型缺少可靠元数据，保留 `unknown`。

HF 卡片标 Apache-2.0，但作者明确说明 AudioCaps 原音频因版权未随包提供，派生音频的
底层媒体权利未在本轮闭合；registry 保持 `unresolved`，不批准训练或正式交付。
可在研究许可后续核对通过后用作单类诊断候选；本轮未运行检测、EER 或阈值选择。

## Doppelganger 环境音研究子集

固定配对表是
[Doppelganger `mseb/test.csv`](https://huggingface.co/datasets/elliottash/doppelganger/tree/eb6170777c1ee5ff68982769b003ab98c3ebf78b/mseb)，
版本 `eb6170777c1ee5ff68982769b003ab98c3ebf78b`。真实 WAV 固定在 FSD50K 镜像版本
`ccf1acaff12f3f4a4c10052dddafbb6d22152c9b`，生成 WAV 固定在
`801be753d84a470a8f94e233e7024a8d306f9842`；逐文件 SHA-256 保存在 provenance。

选择 `ANML/BIRD/FIRE/FOOT/RAIN/WATR/WTHR/WIND` 八类、逐音频为 CC0 的候选；
与 FSD 原始元数据交叉核对 title、uploader、license。398 对候选中剔除 21 对：
明确合成、录音来源不明、事件模仿、不明重传衍生和标注含 Speech 的项。
[source-review.json](source-review.json)记录每项证据和理由；不将“有 Speech 标注”写成
“语音占主导”，也不把疑似合成直接改标 fake。剩余 377 对按每类最多 40 对、
`sha256("42:" + pair_id)` 排序选择 286 对。无听审或逐条独立鉴真；real 标签表示经该
元数据筛查的上游实录来源，仍可能存在未标注的人声或未披露的制作处理。

每条原始 WAV 完整解码，记录原采样率、声道、时长、编码、字节数和哈希。两类均执行
单声道混合、polyphase 重采样到 16 kHz、中心取 64,600 点（4.0375 秒）、峰值归一化到
0.95、写 PCM16 WAV。上游生成程序曾单侧峰值归一化，本轮对两类施加同一最终增益规则，
同时保留处理前 peak/RMS。真实录音短于上游 5 秒条件窗口、生成音频短于最终窗口、静音、
下载/解码失败都逐条登记，连同配对来源一起
退出训练清单；完整 inventory 保持分母。未经处理的 raw WAV 留存用于后续处理条件对照。

此操作控制了文件格式、时长、声道、峰值的直接捷径，不能抹除频带、降噪、采集设备、
生成链或内容差异。尤其 SAO 为音频条件重建，模型效果只覆盖这种生成路径。
原始生成程序使用 5 秒中心裁剪/补零；本轮取同一中心窗口，避免对长真实音频随机取
与生成内容无关的片段。生成器为 Stable Audio Open 1.0，但上游代码未固定模型权重
commit，`generator_version=unknown`；固定已发布 WAV 不等于可逐字节重新生成。

### 许可与归属

- 逐条原媒体为 CC0，归属字段和 Freesound 原网页仍保留；FSD50K 集合许可为
  [CC BY 4.0](https://huggingface.co/datasets/Fhrozen/FSD50k/blob/ccf1acaff12f3f4a4c10052dddafbb6d22152c9b/LICENSE-DATASET.md)。
  引用：Fonseca et al., *FSD50K: An Open Dataset of Human-Labeled Sound Events*。
- 配对表、生成日志及代码由 Doppelganger 作者 Elliott Ash 等按 MIT 提供；生成音频由
  [数据卡](https://huggingface.co/datasets/elliottash/doppelganger)声明为
  [Stability AI Community License](https://stability.ai/community-license-agreement)。
  原许可、作者通知及网页快照在 `source-evidence/`，URL/hash 索引为 `snapshots.json`。
- 当前 registry 的 `approved/training/research` 仅指内部非商业鉴伪研究；不代表正式参赛
  或商业交付已放行。Stability 许可含商业适用条件及禁止用输出训练基础生成模型等条款。
  [FSD 原作者页面](https://zenodo.org/records/4060432)另有比赛用途请联系作者的请求；它不是
  CC BY 条文中的参赛禁令，但正式赛用需由负责人核对。`competition_approved=false`。

本轮未联系外部作者、发布媒体或把带条件的研究登记当成比赛使用许可。

## 划分与泛化协议

本数据来自上游检索基准的 eval/test。这里为新的二分类研究 pilot 重新划分，不能与
FSD50K 或 Doppelganger 标准评测数字比较，也不得继续拿这些媒体声称原基准独立测试。

复用 `assign_splits(seed=42, ratios=(0.7,0.15,0.15))`，按下列关系的连通组分配：

- 同一个 Freesound `source_clip_id` 的真实与生成项；
- 同一 uploader 的全部记录（作为保守采集来源组，不声称 uploader 就是说话人身份）；
- 明确父/派生关联及完全相同的归一化文件 SHA-256。

各 split 真假成对且数量相等，train 训练、dev 早停和模型选择；holdout 只在候选冻结后
评分。本轮对 holdout 仅做必要数据准备和组/字节审计，没有检测器分数。尚未建立跨上传者
的近重复事实，不能宣称未知采集实体都已隔离；新确认的重复边加入现有分组工具后再重划。

来源留出设计：将后续有真实/生成两类的独立环境音来源整体放入开发外部诊断，确保其已知
原媒体/身份/派生组与当前训练数据不相交，并按内容和采集条件检查两类共同支持。最终来源
留出在模型冻结后评估。当前只有一个可用的配对来源，FakeSound 原始录音缺失，因此不能
把“在 FSD real 与 FakeSound fake 上算 EER”当作跨来源验证。

生成器留出设计：以后续加入的完整生成器家族为单位分开发外部诊断和最终 holdout，
同家族版本不得冒充未知家族。其对应真实原媒体及所有派生必须一并离开 train；每个留出桶
保留两类，报告每家族数量、失败和 EER。本轮只有 SAO 家族，无法实测生成器留出 EER。
FakeSound 可在许可核对后补 fake-only 压力测试，但不能用于二类 EER 或选阈值。

## 视频：共同支持缺口与可下载入口

现有 [ViF 受控记录](../vif-control-20261009/README.md)显示宽高/FPS/时长匹配后，
train fake 的 309/323、dev fake 的 73/79 属 HunyuanVideo。其他家族留出多数仅 2–4 条，
两类 dev 为空的桶也存在；留出 Hunyuan 后训练只剩 18 条。底层只有 Panda，不能称为
七生成器或跨来源泛化。另有最终选定的 Hunyuan 对照 train618/dev146，不能混用分母。

新增候选为官方 [ViF-Bench 固定版本](https://huggingface.co/datasets/JoeLeelyf/ViF-Bench/tree/67e11331e26dbf732e0b8c7d4dc52f4442d174dd)：
`source_videos.zip` 为 607,734,233 bytes，上游 LFS SHA-256
`41e79dff9f8ff16f7bcddd99eb18e4019b52f5ed9fe26a6a0fa0739b085907a8`。
本轮实际以 HTTP Range 读取目录及索引：165 Real、2,995 Fake、19 个生成器；
包含 Wan、LTX、SkyReels 等。未下载视频体、未完整哈希或解码，仍是候选入口。

[CVPR 2026 官方补充材料 G 节](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Li_Skyra_AI-Generated_Video_CVPR_2026_supplemental.pdf)
声明 ViF-CoT-4K/ViF-Bench 按 CC BY 4.0 发布，并遵守底层来源许可；
[Panda](https://snap-research.github.io/Panda-70M/)保留非商业研究边界。
`ladm_paths.json` 没有底层来源字段，不能将整个包批准为 Panda 或独立新来源。
与当前 Panda 完整 clip ID 精确交集为 0，但文件名首 11 字符候选组件有 60 个交集；
这仅是候选关系，沿用 `suspected_filename_component:v1` 保守隔离，待 URL/时间戳核对，
不写成已确认重复。该包优先作为外部评估候选，不直接加入现有训练清单。

后续视频任务先把真实 clip 与所有生成派生统一分组，排除任何关联旧冻结 holdout 的
候选组件；按 CogVideoX/Hunyuan/Wan/LTX/SkyReels 家族留出，I2V/T2V 同家族不拆开。
两类统一缩放/FPS/时间窗/重编码，并保留原媒体对照和按条件剔除数量。
只有来源映射闭合且每个来源具备真假两类时，才做来源留出；Panda 与其上游 HD-VILA
不能自动当作独立来源。共同尺寸支持与真实解码仍由后续视频任务验收。

## 复现与接入

在仓库根目录执行，输出路径必须是新目录。共享输入保持只读。

```bash
scripts/devctl run python tools/prepare_fakesound.py \
  --archive /srv/shared-data/datasets/fakesound/FakeSound.zip \
  --output outputs/issue11-fakesound-new

mkdir -p data/issue11-input
curl -fL --retry 3 \
  https://huggingface.co/datasets/elliottash/doppelganger/resolve/eb6170777c1ee5ff68982769b003ab98c3ebf78b/mseb/test.csv \
  -o data/issue11-input/pairs.csv
curl -fL --retry 3 \
  https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/ccf1acaff12f3f4a4c10052dddafbb6d22152c9b/metadata/eval_clips_info_FSD50K.json \
  -o data/issue11-input/source-info.json
scripts/devctl run python tools/prepare_doppelganger.py \
  --metadata data/issue11-input/pairs.csv \
  --source-info data/issue11-input/source-info.json \
  --output data/issue11-doppelganger --workers 4
```

已有实例产物直接复用，不重复下载：将 `data/issue11-doppelganger` 指向团队任务目录的
`doppelganger-v2`，或在自己的配置中使用其实际位置。训练参考
[配置模板](../../../configs/train-logmel-doppelganger.example.json)，路径相对配置文件；
模型任务先按现有协作流程安排 GPU/profile，再决定是否训练。本 Issue 没有启动该训练。
`manifest.jsonl` 是可训练清单，`inventory.jsonl` 包含全部选中样本及排除/失败记录，
不能互换；`sources.jsonl`、`provenance.jsonl`、`audit.json` 与同目录媒体配套读取。

导入工具的行为测试覆盖缺失/损坏记录、许可归属不一致、整对排除、上传者/原媒体隔离、
两类对称处理和禁止覆盖。针对性检查先行，PR 前运行一次 `scripts/devctl check`；
实际验证结果记录于 `results.json`，不把工程检查当作模型效果。
