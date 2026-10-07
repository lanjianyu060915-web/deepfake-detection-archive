# EnvSDD UrbanSound8K 探索子集：固定来源与下载计划

核验与索引日期：2026-10-07。此目录保存的是公开元数据响应和可复算选择脚本，媒体不进入
Git。用途仅为本地非商业研究探索；赛方的外部数据、派生权重和提交规则尚未公开，不能由本
记录推导参赛提交许可。

## 许可与组

- [UrbanSound8K 官方页](https://urbansounddataset.weebly.com/)规定其音频为 CC BY-NC 3.0，
  并要求保留 `FREESOUNDCREDITS.txt` 所列 Freesound 归属。
- [官方命名说明](https://urbansounddataset.weebly.com/urbansound8k.html)定义
  `[fsID]-[classID]-[occurrenceID]-[sliceID].wav`；`fsID` 是原始 Freesound 录音。因此最终
  独立划分的键是 `(UrbanSound8K, fsID)`，`filename` 仅闭合该切片及其生成派生。
- EnvSDD 卡片的 CC BY 4.0 不放宽上述上游 NC 限制。选择文件不下载 DCASE/BBC 或其他真实
  来源。

## 固定输入

- 作者 CSV：`datasplit_dev-d6179d1c.csv`，来自
  `apple-yinhan/EnvSDD@d6179d1c773ca78e4939be5933c9914f97815920` 的
  [`datasplit_dev.csv`](https://raw.githubusercontent.com/apple-yinhan/EnvSDD/d6179d1c773ca78e4939be5933c9914f97815920/datasplit_dev.csv)，
  SHA-256 `cda95f3cafaed74193a6c988904df8de736a48e42ef7b508df47bd0776d8fe0a`。
- `viewer-pages/` 是 Dataset Viewer `train` split、`length=100`、offset 为 0、100、1000、
  5000、10000、15000 的原始只读响应；每页的 SHA-256 被选择脚本固定。请求基址为
  `https://datasets-server.huggingface.co/rows?dataset=EnvSDD%2FEnvSDD&config=default&split=train&offset=<offset>&length=100`。
- 响应中的 `audio.src` 是有期限缓存 URL，只用于本次受控下载。其稳定定位符是
  `(split, row_idx)` 与 Viewer revision `b0832726deb55484dbce869c6f09c41f7e9e5fbb`。

## 精确清单与大小预算

运行 `select_urbansound8k_subset.py` 将从固定页面生成 64 行 manifest：32 个不同 `fsID`，每
个保留一条 real 和同一 `filename` 的一条 `fake/tta/audioldm2`。因此按媒体条目计数严格
均衡为 32 real / 32 fake；所有行附带父 `fsID`，本次只建立一个 exploration 集，不从作者
CSV 的 `usage` 列继承 train/validation 划分。

下载前对 4 个 real/fake 对的公开缓存对象作 HEAD，均为 128,044 B（4 秒、16 kHz WAV）；
预算为 `64 * 128,044 = 8,194,816 B`（7.82 MiB），显著低于 1 GiB。本预算是 HEAD 抽样估计，
实际逐文件字节数和 SHA-256 写入 `data/envsdd-urban/manifest-with-hashes.json` 后才是最终
证据。

该集只含每个所选 `fsID` 的一个切片和一种 fake 路径，故是**探索性**小集，不能据此报告
生成器泛化、完整 `fsID` 家族覆盖或比赛性能。若后续扩大，必须用 CSV 复核同一 `fsID` 的
全部 sibling，并在同一重新划分的 fold 内保留它们。
