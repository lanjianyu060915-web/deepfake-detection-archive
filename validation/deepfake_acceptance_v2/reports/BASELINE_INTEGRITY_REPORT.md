# Deepfake Baseline Integrity Report

| 项目 | 结果 |
| --- | --- |
| Baseline | `face_source_trace_v0.2-dev-deepfake-20260820` |
| 捕获方式 | 只读 SHA-256 与静态配置检查；未导入/运行模型 |
| 结束时核验 | **PASS，50/50** |
| baseline 是否修改 | **否** |

## 冻结范围

- `face_source_trace_v0.2_dev` 的 adapter 三个 Python 文件、`orchestrator.py` 和输入 `config.json`。
- 安装目录主 `深度伪造识别系统.exe`（adapter 实际加载的 PyInstaller PYZ 权威代码载体）、`app.py`、全部 26 个 `src/**/*.py`、runtime config、MobileNet/transform 源码快照和 Haar cascade。
- 空间模型、空间 sidecar/metadata、频域模型/metadata/feature schema/训练溯源、fusion config 和官方 MobileNetV3 backbone 权重。

代码/运行资产清单 36 行，模型/sidecar 清单 10 行；连同 source config 和规范化配置共执行 50 项 size/SHA/配置检查。

## 冻结关键值

- v0.2 `config.json`: `c82b3f0729016c313fd279d7de36623ffb6956fecb7b99aa039b3b1a32dce0fd`
- adapter: `fd74aaa13996f735f9bbbe86afd6d2919c3bee5c0722c019c9f3792be6aeab25`
- native EXE/PYZ: `bc67e52b77fa0559d4d6fe04f024dcdb4a8edf43128f80f7d70947e3da490203`
- spatial model: `5bb4b33936eb3adc631828ea271a466d62070a2b8de1902e5d96122317534794`
- frequency model: `2ec6828eb2a3151784667eb0c105f07cdb41de6ec2df587d28b2766ceee063eb`
- fusion config: `8f76abeabb55da1119b28689e15ccd31eac7f1066e401e766987efbb4603e0b2`
- MobileNetV3 official weight: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`

实际规则保持：`0.62 * spatial_fake_probability + 0.38 * frequency_fake_probability`，图像阈值 `0.36360106194654707`，score 越大越趋 FAKE；CPU、Haar 最大脸外扩 0.25、224×224 和空间/频域预处理均未改变。

最终命令输出：

```text
Baseline: face_source_trace_v0.2-dev-deepfake-20260820
Mode: read_only_sha256_no_model_inference
Checks: 50/50 passed
Status: PASS
```
