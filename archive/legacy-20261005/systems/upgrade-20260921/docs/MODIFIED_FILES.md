# 修改文件清单与原因

对比基线：`D:\刑侦查竞赛\face_source_trace_two_search_v0.1_official`。  
机器生成的逐文件 SHA-256 和文本 unified diff 见 `EQUIVALENT_DIFF.md`。

## 既有文件修改（18 个）

| 文件/组 | 修改原因 |
|---|---|
| `orchestrator.py` | 从双搜索扩展为三独立分支；selected/skipped、统一 deadline、服务 fallback、超时、故障隔离、进度与统一证据结果。 |
| `launcher.py` | 支持 `--modules` 和三模块状态输出。 |
| `config.json` | 新增真实 AIGC 资源、哈希、常驻服务、600 秒底图 timeout、正式外部绝对路径和 8767 端口。 |
| `check_environment.py` | 检查 AIGC adapter/模型/哈希和三模块数据库状态。 |
| `frontend/app.py` | 延续原 Streamlit 页面，新增真实状态、四按钮、上传、进度、服务缓存/降级和三模块结果。 |
| `frontend/components/data_loader.py` | 新增 detection schema、skipped 与旧双模块 task 兼容。 |
| `frontend/components/result_views.py` | 新增 AIGC 页签、证据汇总、不确定语义；禁止融合概率。 |
| `frontend/README.md`、`frontend/TASK_DIRECTORY_CONTRACT.md` | 更新三模块页面和 task 契约。 |
| `frontend/mock_task/README.md` | 标明 AIGC mock 仅供布局测试。 |
| `run_app.bat` | 统一转发到正式 `start.bat`，避免旧脚本硬编码 8766。 |
| `tests/test_orchestrator_contract.py` | 同步三模块 contract 与 disabled 语义。 |
| `tests/README.md`、`tests/fixtures/README.md` | 增加三模块、服务和真实长回归说明。 |
| `README.md` | 改写为 v0.2 正式运行、模型、架构、启动和迁移说明。 |
| `CHANGELOG.md` | 记录三模块正式变更。 |
| `KNOWN_ISSUES.md` | 如实记录源码缺失、CPU、闭集、无持久底图 index 和外部依赖。 |
| `docs/FRONTEND_VERIFICATION.md` | 用 8767 真实浏览器端到端验收替换旧双搜索记录。 |

## 主要新增代码

- `modules/deepfake_detection/`：三份冻结适配器、AIGC 常驻 worker 和说明。
- `modules/identity_search/persistent_worker.py`：复用真实 buffalo_l 模型对象和身份缓存。
- `runtime_services.py`：两个隔离 worker 的 JSON 行协议、健康状态、请求锁、deadline 和清理。
- `system_status.py`：模型/适配器哈希、真实图库、身份缓存 fingerprint/NPZ 和实际设备检查。
- `frontend/components/status_adapter.py`：把真实系统快照映射为 UI 可用/空/缺失状态。
- `tests/test_runtime_services.py`、新增 orchestrator/frontend 测试：覆盖服务协议、禁用、超时、故障隔离和兼容。
- `tests/run_real_integration.py`：显式真实模型长回归入口，不进入轻量 mock 套件。
- `tools/generate_equivalent_diff.py`：无 Git 目录的可复现 SHA-256/unified diff 生成器。

## 主要新增交付材料

- `start.bat`、`start.ps1`、`requirements*.txt`。
- `examples/`：AIGC 与三模块真实样例。
- `docs/`：源码定位、依赖、数据库、模型、架构、测试、运行状态和差异报告。
- `test_results/`：成功/失败真实 JSON、常驻服务摘要、轻量测试日志和 UI 截图。
- `screenshots/three_module_*.png`：真实页面关键状态截图。
- `runtime/cache/deepfake/.../mobilenet_v3_small-047dcff4.pth`：固定官方基础权重，防止运行时联网。

## 删除

没有删除基线中的业务文件。`run_app.bat` 是同路径内容替换，不计删除。正式 release 会有意不分发继承的 v0.1 `packaging/dist`、`packaging/stage`、`packaging/build`、`release_v1.0` 和开发运行时 task；它们不是 v0.2 业务程序，排除原因会写入发布清单。

