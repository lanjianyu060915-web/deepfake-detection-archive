# v0.1.0 双分支离线基线

本版提供带权重、CPython 和 Python 依赖的音频/视频检测系统。输入本地无标签媒体目录，
输出连续伪造分数、覆盖率与逐样本失败，支持中断恢复。运行时不安装依赖或下载模型。
这是可运行的非商业研究基线；音频默认是环境音 AST，视频默认是 ResNet18 部分微调。
语音、音乐和跨生成器效果继续优化，官方输入输出接口尚待适配。

## 下载与启动

下载 v0.1.0 的全部 `deepfake-v0.1.0-linux-x86_64.tar.gz.part-*` 和 `SHA256SUMS`，
放在同一目录。校验一次、合并解压：

```bash
sha256sum -c SHA256SUMS
cat deepfake-v0.1.0-linux-x86_64.tar.gz.part-* | tar -xz
cd deepfake-v0.1.0
./run.sh --help
./run.sh predict --config configs/audio.json --input media/audio --output runs/demo-audio
./run.sh predict --config configs/video.json --input media/video --output runs/demo-video
```

目标是 Linux x86_64、glibc 2.35 或兼容环境。包内提供 CPython 3.12 与 CUDA 13.2
构建的 PyTorch；GPU 推理需要兼容驱动，本版实测 RTX4090 / 595.91.07。
系统 PATH 需有 FFmpeg 和 ffprobe（支持 `-fps_mode`），两者不随包分发。
当前团队服务器已全局安装。CPU 可用 `--device cpu`，无需另装 Python 或 pip。
首次解压约需8GB，另为输入与评分输出留空间；不支持 Windows/macOS。

## 对实际媒体评分

音频、视频按两个赛项分别调用，不读取视频音轨作为音频结果：

```bash
./run.sh predict --config configs/audio.json --input /data/audio --output /results/audio
./run.sh predict --config configs/video.json --input /data/video --output /results/video
# 中断后，使用同一输入、模型和输出目录
./run.sh predict --config configs/video.json --input /data/video --output /results/video --resume
```

支持 WAV/FLAC/MP3/M4A/OGG/AAC/OPUS 与 MP4/MKV/AVI/WEBM/MOV/MPEG/M4V。
递归读取对应格式文件，`sample_id` 是输入根目录内的相对路径，保留扩展名、中文和子目录。
音频通过 FFmpeg 转为16kHz单声道；当前音频模型取均匀位置的单个4秒片段，短音频循环补齐。
视频采样4帧，按 checkpoint 的预处理与聚合评分；这些是固定基线参数，不代表完整时序建模。

每次使用新输出目录；中断续跑加 `--resume`。输入或模型发生变化时开新目录。
输出目录和输入目录分开放置。无标签输入不计算 EER。

| 文件 | 用途 |
| --- | --- |
| `scores.csv` | 所有样本成功时生成，列为 `sample_id,score`；分数越高越伪造，通常为 logit，不是概率 |
| `scores.jsonl` | 每个输入的分数或明确失败；不编造兜底分数 |
| `report.json` | 样本数、覆盖率、失败原因、耗时与吞吐 |
| `manifest.jsonl` / `inference.json` | 输入清单与本次实际解析配置 |
| `metadata.json` | 固定输入和模型的恢复身份 |

有媒体失败时返回退出码2，保留详情并不输出不完整的 `scores.csv`；修复输入后用新目录运行。
恢复时若输入或配置有冲突，拒绝继续并保留原运行产物；已有 CSV 仍属于原运行，不能当作新输入结果。
这是项目本地输出约定，不能把 CSV 直接标成已确认的官方提交。
正式列名、分数方向和输入 ID 规则在平台样例明确后集中修改 adapter。

## 开发与重建

源码 checkout 使用共享环境运行同一入口：

```bash
scripts/devctl run python -m deepfake_detection_archive predict \
  --config /path/to/deepfake-v0.1.0/configs/audio.json \
  --input /data/audio --output outputs/audio
```

维护者在对应tag的源码checkout中，用已取得的本地权重与环境构建，
不自动下载或读取训练/holdout媒体：

```bash
scripts/devctl run python tools/build_release.py \
  --output outputs/releases/deepfake-v0.1.0 \
  --site-packages /opt/venvs/deepfake-cu132/lib/python3.12/site-packages \
  --python-home /opt/uv/python/cpython-3.12.15-linux-x86_64-gnu \
  --audio-weight /path/to/urban-candidate/training/best.pt \
  --video-weight /path/to/video-candidate/training/best.pt
tar -C outputs/releases -czf outputs/releases/deepfake-v0.1.0-linux-x86_64.tar.gz deepfake-v0.1.0
split -b 1900M -d outputs/releases/deepfake-v0.1.0-linux-x86_64.tar.gz \
  outputs/releases/deepfake-v0.1.0-linux-x86_64.tar.gz.part-
```

包内 `release.json` 标识构建源码、平台和权重，`deps/uv.lock` 固定 Python 依赖；
依赖快照保留构建环境已有工具，并非最小容器。目录搬移后不依赖原仓库、训练路径或共享虚拟环境。
来源、版权与非商业限制见包内 `licenses/README.md`，项目内见 `docs/release-licenses.md`。
