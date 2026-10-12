# 离线双分支运行包（Issue #31）

可搬移的“音频 AST 研究候选 + 视频 layer4+头微调 seed42”双分支离线运行包：
直接复用现有 CLI、`submission/runner.py`、`submission/package.py` 与
`submission/adapter.py`，不新增服务、UI 或框架，也不把视频音轨并入音频任务。

## 1. 候选身份

| 分支 | 架构 | 包内权重 | 字节 | sha256 |
| --- | --- | --- | --- | --- |
| 音频 | `audio_ast_v1` | `weights/ast-mixed12-class-balanced-seed42.pt` | 342,935,981 | `4b1da5d98caebd7d47eb86033c312a4c9699a7e6f8ec13e7ddca72efbcc884f7` |
| 视频 | `video_frame_finetune_v1` | `weights/video-l4head-seed42.pt` | 44,919,043 | `16abc4e9828f73545d60da874a0da16b2a394c63c465896c9ab7e5c5bf465669` |

音频候选：音频 AST（Audio Spectrogram Transformer，12 层、`hidden_size=768`），
由 `audio-ast-20261011` 的 AudioSet 预训练权重微调，训练数据为 AIME 音乐 +
EnvSDD 城市环境音，13888 train / 96 dev，`balanced_sampling=true`、seed 42，
`best_dev_eer=0.0`、`holdout_accessed=false`；许可为研究用途
（AIME：CC-BY-4.0 AND CC-BY-NC-SA-4.0；EnvSDD：CC-BY-4.0 AND CC-BY-NC-3.0）。

视频候选：`resnet18` 主干，仅解冻 `layer4` 并接分类头（`hidden_dim=64`）的
逐帧微调权重。该权重 sidecar 的 `config.encoder_checkpoint` 仍记录训练期的
编码器绝对路径，但 `video_frames` 推理路径从不加载它；这一行为由回归测试
`tests/test_offline_dual_branch.py::test_video_branch_relocates_and_ignores_training_encoder_path`
固化，因此迁移后无需该训练期文件。

两个权重的包内 sidecar（`*.pt.json`）与 `weights/ast-experiment.json`
一并随包，用于核对来源与哈希；`configs/*.json` 的 `checkpoint_sha256`
与上表一致，权重被替换或损坏时运行直接失败。

身份澄清：视频分支要的是 `layer4+头` 微调权重（44.9 MB，
`video-finetune-20261011/seed42`，见 `docs/experiments/video-finetune-20261011/`），
而非同名的仅头部权重（约 0.14 MB）；两者 sha256 不同，本包使用前者。
音频分支同理确认为音频 AST 音乐/环境音混合候选（`audio_ast_v1`），
而非其他 logmel/SSL 候选。

## 2. 包布局（全部相对路径，可整目录搬迁）

```
offline-package/
  configs/        # sample-*/batch-* 四份配置（data_root 均相对本目录）
  manifests/      # batch/sample × audio-{env,music,speech} / video-{control,vifbench} / *-dev
  media/          # 自带极小样例媒体（约 3.0 MB / 30 个文件），完全离线可跑
  weights/        # 两个真实权重文件 + sidecar（非软链接）
  deps/           # uv.lock + 依赖说明
  shared-data ->  # 相对软链接，指向团队只读素材根；批量配置经它读取共享媒体
  SHARED-DATA.md  # 迁移后如何重挂 shared-data
```

迁移后不依赖任何原训练绝对路径或原预训练文件位置：包内只用相对路径，
`sample-*` 配置自带媒体，`batch-*` 配置只经 `shared-data` 挂载点访问共享媒体
（可重指向新机器素材根）。

## 3. 使用（中文）

```bash
# 自带样例，完全离线（无需 shared-data）
scripts/devctl run python -m deepfake_detection_archive infer \
  --config offline-package/configs/sample-audio-dev.json
scripts/devctl run python -m deepfake_detection_archive infer \
  --config offline-package/configs/sample-video-dev.json

# 批量（需先挂好 shared-data）
scripts/devctl run python -m deepfake_detection_archive infer \
  --config offline-package/configs/batch-audio-dev.json
scripts/devctl run python -m deepfake_detection_archive infer \
  --config offline-package/configs/batch-video-dev.json
```

长任务在 tmux 中运行；中断后用同一配置追加 `--resume` 续跑（见第 6 节）。
包内不放权重/大媒体/日志等大产物；`runs/`、`shared-data` 的实体均在包外。

## 4. 数据与选取规则（仅 dev，未触碰 holdout）

| 桶 | 样本数 | real/fake | 选取规则 |
| --- | --- | --- | --- |
| audio-env | 200 | 100/100 | dev 行，按 `sample_id` 排序后 real/fake 各取前 100 |
| audio-music | 96 | 16/80 | dev 行全量（AIME 音乐，real 稀少） |
| audio-speech | 200 | 100/100 | real 前 100 + 100 个 fake，按生成器轮转均衡 |
| audio-dev | 496 | 216/280 | 上述三桶合并 |
| video-control | 146 | 73/73 | dev 同域控制集全量 |
| video-vifbench | 190 | 95/95 | 每个 ViF-Bench 诊断配对取 5 real + 5 fake（real 去重） |
| video-dev | 336 | 168/168 | 上述两桶合并 |
| sample-audio-dev | 18 | 9/9 | 每桶 real/fake 各 3 |
| sample-video-dev | 12 | 6/6 | 每桶 real/fake 各 3 |

所有清单行 `split == "dev"`；生成工具 `tools/prepare_offline_package.py`
中的 `check_no_holdout` 在选取阶段显式拒绝任何 `split == "holdout"` 行。

新生成器泛化按 `docs/experiments/video-generalization-20261011/` 的
ViF-Bench 清单与诊断配对补充评测（第 5.3 节）。

## 5. 结果

### 5.1 正常环境 vs 离线迁移（同一批媒体逐样本对比）

| 运行 | 样本 | 耗时 | 吞吐 | 峰值自用显存 |
| --- | --- | --- | --- | --- |
| audio-dev 正常环境 | 496 | 28.20 s | 17.59 样本/s | 0.373 GiB |
| audio-dev 离线迁移 | 496 | 38.10 s | 13.02 样本/s | 0.373 GiB |
| video-dev 正常环境 | 336 | 83.60 s | 4.02 样本/s | 0.099 GiB |
| video-dev 离线迁移 | 336 | 91.53 s | 3.67 样本/s | 0.099 GiB |

逐样本分数对比（`tools/diff_scores.py`）：`max_abs_delta = 0.0`，
`exact_equal_count` 分别为 496 / 336，`status_mismatch_count = 0`，即离线迁移与
正常环境给出逐样本完全一致的分数（样例集 18 / 12 亦为 0.0）。两次运行均在
同一共享卡上、他机任务占用约 0.4–0.8 GiB 时测得；更早一次正常环境运行在
他机占用约 17.4 GiB 时耗时为 audio 91.19 s、video 175.36 s，说明吞吐主要
受共享负载影响，与离线/在线无关。本任务自用峰值显存始终 ≤ 0.4 GiB。

### 5.2 分桶 EER / 覆盖率 / 失败数

| 桶 | 样本 | EER | coverage | failure_count |
| --- | --- | --- | --- | --- |
| audio-env | 200 | 1.00% | 1.0 | 0 |
| audio-music | 96 | 0.00% | 1.0 | 0 |
| audio-speech | 200 | 31.00% | 1.0 | 0 |
| audio-dev（合并） | 496 | 13.8889% | 1.0 | 0 |
| video-control | 146 | 16.44% | 1.0 | 0 |
| video-vifbench | 190 | 36.84% | 1.0 | 0 |
| video-dev（合并） | 336 | 28.5714% | 1.0 | 0 |

样例集：audio 18 → 11.11%，video 12 → 16.67%。正常环境与离线迁移的分桶
EER 完全一致，失败数为 0。音频 speech 桶明显高于 env/music（语音伪造更难），
视频新生成器桶（vifbench 36.84%）高于同域控制桶（control 16.44%），符合预期泛化落差。

### 5.3 新生成器泛化（ViF-Bench 诊断配对）

同域控制集（control）取自 ViF-Panda 研究媒体 `dev146`：73 对 HunyuanVideo
伪造与配对真实，EER 16.44%。

新生成器按 `docs/experiments/video-generalization-20261011/` 的 ViF-Bench
清单与其 `diagnostics/` 配对（每个生成器用同一组配对真实的 103/104 对）
补跑全量 1949 个样本（104 real / 1845 fake，均 `split == "dev"`），
整体 EER 31.33%、coverage 1.0、failure_count 0，508.75 s（3.83 样本/s）。
逐生成器 EER（样本数 = real+fake 配对后计数，含共用真实池）：

| 生成器 | n | EER |
| --- | --- | --- |
| SkyReels-V2 | 208 | 16.35% |
| HunyuanVideo | 206 | 16.50% |
| pika-v2 | 180 | 16.67% |
| pixverse-v4-5 | 184 | 19.57% |
| LTX-Video-13B-T | 200 | 20.00% |
| Wan2.1-T2V-1.3B | 208 | 20.19% |
| hailuo | 162 | 23.46% |
| kling-v1 | 160 | 27.50% |
| Wan2.2-T2V-14B | 208 | 27.88% |
| Wan2.2-TI2V-5B-T | 208 | 27.88% |
| sora-2 | 188 | 35.11% |
| HunyuanVideo-I2V | 208 | 35.58% |
| CogVideoX1.5-5B-T | 208 | 33.65% |
| Wan2.1-VACE-1.3B-T | 208 | 36.54% |
| LTX-Video-13B-I | 204 | 42.16% |
| gen4-turbo | 128 | 45.31% |
| Wan2.2-I2V-14B | 208 | 48.08% |
| SkyReels-V2-I2V-14B-540P | 206 | 49.51% |
| Wan2.2-TI2V-5B-I | 208 | 53.85% |

结论：图生视频/新版本生成器（Wan2.2-I2V/TI2V-I、SkyReels-I2V、LTX-I）
明显更难，文生视频与旧版本相对容易；同域 HunyuanVideo 控制在 16.4%，
跨生成器整体退化到 31.3%。

## 6. 中断恢复

- 视频：运行中向真实进程发 SIGKILL，已写 217/336 行；用同一配置加
  `--resume` 续跑，本会话新增 119 行，最终 336 行完整，EER 28.5714% 不变。
- 音频：tmux 前台运行中发送 Ctrl-C（SIGINT），已写 118/496 行；加 `--resume`
  续跑，本会话新增 378 行，最终 496 行完整，EER 13.8889% 不变。

resume 依赖 `.runner.lock` 与逐行追加的 `scores.jsonl`，已完成的样本不重算，
报告仍覆盖全量。

## 7. 离线/禁用隐式下载的验证方法

本机 `unshare -n` 不可用，改用等价手段：

1. 将包整目录复制到独立位置，运行目录无关（`cwd=/`），只用仓库内 CLI 与共享虚拟环境；
2. 进程内安装 CPython audit hook，拦截 `socket.connect`、`socket.getaddrinfo`、
   `urllib.Request`、`http.client.connect` 等出网事件并直接抛错；
3. 同时设 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、
   `PYTHONNOUSERSITE=1`。

证据：探针调用 `socket.getaddrinfo("huggingface.co", 443)` 与
`socket.create_connection(("1.1.1.1", 443))` 均被
`offline guard blocked network event` 拒绝；在同样条件下四条样例/批量推理
均 `exit=0`、coverage 1.0、failure_count 0。权重随包为真实文件，
`checkpoint_sha256` 固定，保证不隐式回退到预训练下载。

## 8. 局限

- 单 seed（seed42），未做多种子方差；语音仅作诊断，不并入音频任务，也不融合音轨。
- 音频音乐候选 dev EER 0.00%，说明该 dev 划分偏易，跨集泛化未覆盖。
- 新生成器只覆盖 ViF-Bench 一份基准，且受共享 GPU/CPU 波动（load 约 12–16）影响，
  吞吐为当时实测值。
- 许可为研究用途，正式提交前需按赛题许可要求复核。

## 9. 复现

设 `SHARED_ROOT` 为团队只读素材根（本机挂载点），包生成：

```bash
export SHARED_ROOT=<团队共享素材根>
scripts/devctl run python -u tools/prepare_offline_package.py \
  --package outputs/offline-dual-branch-20261012/offline-package \
  --shared-root "$SHARED_ROOT" \
  --config-dir docs/experiments/offline-dual-branch-20261012/configs \
  --audio-weight "$SHARED_ROOT/deepfake-audio-music-20261011/mixed12-class-balanced-seed42/training/best.pt" \
  --video-weight "$SHARED_ROOT/lanjianyu-issue21-20261011/outputs/video-finetune-20261011/seed42/training/best.pt" \
  --lock uv.lock \
  --env-manifest "$SHARED_ROOT/deepfake-audio-expand-20261011/envsdd-urban/manifest.jsonl" \
  --env-root "$SHARED_ROOT/deepfake-audio-expand-20261011/envsdd-urban" \
  --music-manifest "$SHARED_ROOT/deepfake-audio-music-20261011/aime/manifest.jsonl" \
  --music-root "$SHARED_ROOT/deepfake-audio-music-20261011/aime" \
  --speech-manifest "$SHARED_ROOT/lanjianyu-issue9-20261011/data/main/manifest.jsonl" \
  --speech-root "$SHARED_ROOT/lanjianyu-issue9-20261011/data/LA" \
  --video-control-manifest "$SHARED_ROOT/lanjianyu-issue20-20261011/outputs/experiments/video-generalization-20261011/inputs/dev146.jsonl" \
  --video-control-root "$SHARED_ROOT/lanjianyu-issue10-20261011/vif-panda-research/media" \
  --vifbench-manifest "$SHARED_ROOT/lanjianyu-issue20-20261011/outputs/experiments/video-generalization-20261011/vifbench/manifest.jsonl" \
  --vifbench-root "$SHARED_ROOT/lanjianyu-issue20-20261011/outputs/experiments/video-generalization-20261011/vifbench/media" \
  --vifbench-diagnostics "$SHARED_ROOT/lanjianyu-issue20-20261011/outputs/experiments/video-generalization-20261011/vifbench/diagnostics"
```

重建后 `manifests/` 14 份、`configs/` 4 份与样例媒体与本包逐字节一致。

行为回归：

```bash
PYTHONPATH=src scripts/devctl run python -m pytest -q tests/test_offline_dual_branch.py
```
