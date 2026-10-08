# EnvSDD UrbanSound8K：固定 CSV 完整来源组研究集

状态：**530/530 原始 WAV 全量字节、元数据和独立 PCM 复验通过，归属缺口仍保留。**
结果见 `pcm-recovery-verification.json` 与最新 `verification-*.json`；未训练。

## 冻结范围

固定 Hub revision `b0832726deb55484dbce869c6f09c41f7e9e5fbb`。
作者 CSV、33 页来源元数据及各自 SHA-256 均记录在 `selection.json`；
清单 SHA-256 为 `554f77b083bbc3c649516c37045064c27e952147d675c6e3125cc396d79749a6`。

只选择 Viewer 中 filename 集与固定作者 CSV 同 fsID 的全部 filename 集相等，
且每个 filename 有 real 与至少 3 个 fake generator 的组，按数值 fsID、filename、row_idx 排序。
结果固定为 **40 fsID、106 filenames、530 条**：106 real / 424 fake。
每个 filename 有 5 条：real、audiogen、audioldm1 的两行、audioldm2。
Viewer 的两条 audioldm1 使用相同 `generative_model` 字符串；保留各自 row_idx，
不凭该元数据给它们编造更细的版本标识。

“完整”只指固定作者 CSV 中这些 fsID 的 filename 家族；不表示收齐原始 Freesound 录音的
所有切片、全部 UrbanSound8K、全部 EnvSDD split 或所有生成器派生。
包含 9 类 event，未选到 gun_shot；类分布与媒体真伪数量均不平衡。
未继承作者 CSV `usage` 划分，未建立本地 train/dev/test。

## 获取与复验

- `PLAN.md`：媒体下载前的限定计划。
- `selection.json` / `selection-lock.json`：不可重选的清单及哈希。
- `acquire.py`：`freeze` 固化；`fetch` 仅恢复同清单。25 个对应 100 行页重新请求，强制核对
  `x-revision`、完整非 audio 元数据、Urban-only 和固定 revision/row 的音频路径。
- `run-*/`：刷新页面原文、逐页/逐媒体 journal、完成报告；失败尝试保留。
- `verify.py`：独立离线重算选样、原页/刷新页哈希与 metadata、每条完整 WAV 解码与 body SHA，
  检查缺失/重复/额外媒体和诊断统计。
- `test_acquire.py`：6 项本地边界测试，无网络/模型。

媒体在 Git 忽略的 `data/envsdd-urban-full-family/audio/<fsID>/<row_idx>.wav`。
最多 3 并发、45 秒超时、每条最多 3 次尝试，每 body 至多 256,088 字节。
530×3×256,089 的请求读取预算为 407,181,510 字节，低于 1 GiB。
媒体完整解码为 16 kHz mono PCM16 / 64,000 帧后才原子发布；记录完整 body SHA-256、
RMS、peak、all_zero、zero_fraction、clipping_fraction。截断或异常逐行记账，不静默丢样本。
旧 `data/envsdd-urban/` 64 条与其历史证据未修改。

## 来源、归属与使用边界

UrbanSound8K 编纂者为 Justin Salamon、Christopher Jacoby、Juan Pablo Bello，
采用 [CC-BY-NC-3.0](https://urbansounddataset.weebly.com/)；
[官方命名约定](https://urbansounddataset.weebly.com/urbansound8k.html)定义 fsID 为原始 Freesound 录音。
每条清单保留 `https://freesound.org/s/<fsID>/` 归属定位链接及作者 CSV 字段。
EnvSDD 数据卡的 CC-BY-4.0 不放宽上游 NC 限制。

`license-sources/` 保存两张官方页面与固定 revision 数据卡原文、来源 URL 和哈希。
原始 `FREESOUNDCREDITS.txt` 尚未取得，逐录音作者归属核验未完成。
官方 [Zenodo 固定记录](https://zenodo.org/records/1203745)只列 6,023,741,708 字节完整压缩包；
本轮只探测其前 1 MiB，HTTP 206/Content-Range 已记录，前缀未包含 credits。
该前缀留在忽略数据目录 `source-excerpts/`，未抽取其中任何媒体，未声称通过整包 MD5。
`official-credits-probe.json` 保留有限探针和不足原因。

因此本批只作为本地非商业研究数据治理证据，**归属尚未完整验收，不进入训练或正式比赛使用**。
未训练、未运行模型、未产生 EER、未分组划分、未向外上传或提交。

## 复现检查

```bash
.venv/bin/ruff check docs/research/source/envsdd/full-family-20261008/*.py
.venv/bin/ruff format --check docs/research/source/envsdd/full-family-20261008/*.py
.venv/bin/mypy --strict docs/research/source/envsdd/full-family-20261008/*.py
.venv/bin/python -B -m unittest discover -s docs/research/source/envsdd/full-family-20261008 -p 'test_*.py' -v
.venv/bin/python -B docs/research/source/envsdd/full-family-20261008/verify.py
```

## 2026-10-08 实际格式恢复

原进程中断后306条均有已验证账本；首轮续传以原PCM16限定运行，450成功、80条有效PCM24明确失败，原记录不改写。新acquire_pcm.py保留原来源字节并支持原生8/16/24/32位PCM，格式计划与red/green边界测试见本目录；verify_pcm.py用独立libsndfile解码全530条，与完整原始body/hash及全部meta对齐。

最终72,989,612字节、530唯一hash、0全零信号、48条含削波样值；削波仅记账不自动剔除。real位深16/24为90/16，fake为360/64，每个generator比例相同；当前没有位深边际真假不对称证据，不能据此排除其它来源/量化捷径。来源CSV范围40fsID/106文件完整，无train/dev/test或模型分数。

原始credits正在另一个有6.1GB来源预算的[独立获取分支](../../urbansound8k-attribution-20261008/PLAN.md)恢复，未取得前仍不训练/比赛使用；该分支不改写本次530条获取预算或清单。
