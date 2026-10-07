# ViF Panda 冻结 ResNet18 探索 pilot

本目录保存分数产生前的预注册、两份固定训练配置、准备脚本和审计记录。
仅限本地非商业计算研究；数据/基础权重的来源许可不代表赛方已批准外部资源。
此 pilot 不构成 P1/P2 正式通过，不报告独立组置信区间。

目标为约 1,024 条媒体，使用完整真实/生成配对和保守关系组件取样，
7 个生成器按来源占比分层。保留 parent、SHA-256 和疑似文件名前缀的传递闭包。
审计采用 4 个均匀全帧、224×224 保比例补边、固定本地 ResNet18 权重，
不裁人脸。特征均值余弦和逐帧 pHash 仅生成防泄漏候选；
超过预声明阈值的候选一律保守合并，不将其冒充身份或确定的同源关系。
`relationship-preservation.json` 保留原 derivation 标记，最终清单以合并后的
保守组件作 `derivation_group_id`，`identity_id/group_id/original_id` 仍为 unknown。

`preregistration.json` 固定规则及 seed42/43 配置哈希。两份配置除随机种子和
独立输出目录外相同：512→64→1 头、mean logits 聚合、batch64、bf16、
AdamW learning rate 0.001、至多 20 epoch、dev EER patience5。
原注册不因工程审阅修复而重写，修订记录解释改变及验证。

执行顺序（仓库根，已安装当前 cu130 环境）：本次 `freeze` 已执行。
新克隆中 `outputs/` 受 Git 忽略，先将原三份冻结文件复制到工作目录；
存在同名字节不一致的文件则停止，不能覆盖或重新注册本次实验。

```bash
.venv/bin/python - <<'PYSETUP'
from pathlib import Path
source = Path("docs/research/source/vif-pilot")
work = Path("outputs/experiment-planning/vif-pilot")
work.mkdir(parents=True, exist_ok=True)
Path("cache/vif-pilot").mkdir(parents=True, exist_ok=True)
for name in ("preregistration.json", "train-seed42.json", "train-seed43.json"):
    original = (source / name).read_bytes()
    target = work / name
    if target.exists() and target.read_bytes() != original:
        raise RuntimeError(f"已有工作文件与冻结原件不同: {target}")
    if not target.exists():
        target.write_bytes(original)
PYSETUP
PYTHONPATH=src .venv/bin/python docs/research/source/vif-pilot/prepare_pilot.py select
PYTHONPATH=src .venv/bin/python docs/research/source/vif-pilot/prepare_pilot.py decode
PYTHONPATH=src .venv/bin/python docs/research/source/vif-pilot/prepare_pilot.py features
PYTHONPATH=src .venv/bin/python docs/research/source/vif-pilot/prepare_pilot.py lock
```

`select` 要求完整源 ZIP 的导入清单已发布且哈希一致，拒绝部分下载。
`decode` 并行 4 个 CPU 工作线程；`features` 需要 CUDA，只做冻结编码器前向。
两个阶段写出逐样本/批次失败账；任何解码或表征错误都会阻止锁定。
`duplicate-audit.json` 和组件体积记录在划分前发布，即使巨组件导致划分失败也可审阅。
`split-lock.json` 存在且校验通过后才可启动训练；不改变阈值来迎合划分或成绩。

主代理协调单 GPU 后，使用项目现有训练入口：

```bash
PYTHONPATH=src .venv/bin/python -m deepfake_detection_archive train --config outputs/experiment-planning/vif-pilot/train-seed42.json
PYTHONPATH=src .venv/bin/python -m deepfake_detection_archive train --config outputs/experiment-planning/vif-pilot/train-seed43.json
```

训练配置保留本次机器的实际绝对路径。迁移机器时应生成明确的新路径版本并登记新哈希，
保留本目录原始配置；不得声称改路径后的字节仍等于原注册。
真实训练结果应另存运行目录；本准备脚本不执行训练、不产出鉴伪分数。
训练入口仅消费 train/dev 的媒体，holdout 只能用于冻结后一次验收；
预划分的全媒体解码和近邻检查属于数据治理，不能据此调模型或选择阈值。

已知限制：稀疏帧与对齐 pHash 可能漏掉时间偏移/局部副本，ImageNet 特征近邻可能
只反映语义相似。审计只覆盖取出的 pilot 媒体，未知关系仍可能存在。
分辨率、帧率、编码和时长按真假及划分列分布，任何强关联都应标为捷径风险。
单一 Panda 子集的探索结果不能外推为未知生成器/跨来源或真实比赛能力。
