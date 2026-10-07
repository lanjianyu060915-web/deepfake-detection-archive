# 已暂停：真实模型演练草稿尚未验收

用户明确暂停后保存。当前没有真实训练 checkpoint，未运行实际 detector 的
infer/evaluate/resume，未启动模型打包或篡改测试。既有 CPU wheel 获取、干净断网
安装及基础 CLI 证据保持原字节。

## 当前草稿

- `source/model_rehearsal.py`：已有 dev-only 12 真/12 假固定 ID 选样、媒体/权重复制验哈希、
  两次完整推理比对、精确自有子进程 SIGSTOP/SIGINT/SIGCONT、中断恢复和再次 resume
  幂等检查框架。仅完成先前 Ruff/格式检查；**独立审阅仍有未修复问题，不得执行。**
- `outputs/experiment-planning/offline/model-job.pending.json`：显式占位模板，
  `training_completed=false`，未绑定真实 checkpoint。
- 已结束的单行合成写入进程探针仅证明 OS 停止/信号机制，不能作为模型验证。
  记录在 `outputs/experiment-planning/offline/signal-probe/report.json`。

## 未修复审阅项（暂停时尚未开始应用修复）

1. **训练凭据强绑定**：当前仅信 job 布尔及任意 provenance 文件。需要验证真实
   `experiment.json` 和 `training/run.json` 均 `status=complete`，两份 checkpoint
   SHA-256 与实际 best.pt 一致且不同于 initial checkpoint，best_epoch 一致；
   `training/history.json` epoch 数与 epochs_completed 一致，best_epoch 及之前累计
   optimizer_updates > 0。训练器 provenance、train/dev 计数、来源文件哈希和清单规范哈希
   必须相互一致；dev 输入必须精确对应真实训练清单的 dev 成员。
2. **固定环境与 job 完整性**：需要把既有 CPU 安装报告/项目 wheel SHA、实际
   sys.prefix、包 site-packages 归属、torch CPU 版本及源码对 wheel 成员的逐项哈希
   校验绑定到 preparation；`job.json` 自身也要纳入摘要并在执行前复核。
   Python argv 不应 resolve 为系统解释器而丢失 venv。当前草稿还没有这些强绑定。
3. **失败有界与留证**：namespace 启动和常规 infer/evaluate 命令尚无 timeout；
   SIGTERM 清理等待超时后需要对同一 Popen 自有进程 kill/reap。失败状态及异常必须
   写入报告，不能只留下未完成记录或挂住。
4. **中断前缀证据**：应持久化实际 `/proc/<owned_pid>/status` 停止状态；除检查
   resume 保持 after-exit 前缀，还要验证 after-exit 完整前缀保留 SIGSTOP 时的完整行。

独立审阅者已确认可复用的公开接口（无需重写哈希算法）：

```python
from deepfake_detection_archive.data.manifest import (
    load_manifest, manifest_hash, load_sources,
)
from deepfake_detection_archive.tracking import code_identity
```

- `load_manifest(Path) -> list[SampleRecord]`；`record.to_dict()` 可精确比较 dev 成员。
- `manifest_hash(records)` 复现规范清单哈希；train/dev 过滤复现 data_sha256。
- `load_sources(Path)` 验证来源登记结构；sources_sha256 仍是来源文件字节 SHA-256。
- `code_identity()` 给出 code_sha256 及可能存在的 commit/dirty。代码身份从安装包路径
  向上寻找 Git；当前环境在仓库内，运行前须由 root 协调提交/受跟踪写入冻结，恢复验证后释放。
  不手工覆盖或伪造代码身份字段。

## 恢复入口

1. 用户恢复后先完成上述修复和独立复审，归档最终脚本精确字节。
2. 与 ASV 执行分支对接真实 LogMel42 完成信号、best.pt hash、dev-only 清单、sources、
   media root、推理配置及 experiment/run/history/许可证明。
   ASV 当前暂停记录：`docs/experiments/asv-pilot-20261007/PAUSED.md`；暂无真实 checkpoint。
3. 更新待执行 job 并先做 prepare，核对真实资产绑定；协调 Git 状态短暂冻结后再 execute。
4. 追加独立模型演练报告，保留旧安装证据，不宣称安全屋或代表性 500 条验收。

暂停时本分支没有后台下载、安装、训练或推理进程。下载与安装任务已结束；唯一进程控制
探针的自有子进程已退出（退出码 -2）。未操作 Git，未修改主动算法。
