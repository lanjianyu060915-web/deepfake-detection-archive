# v0.2 正式发布清单

正式目标：`D:\刑侦查竞赛\releases\face_source_trace_three_module_v0.2_official`  
生成来源：`D:\刑侦查竞赛\face_source_trace_three_module_v0.2_dev`  
基线：`D:\刑侦查竞赛\face_source_trace_two_search_v0.1_official`

## 正式版包含

- 三模块编排、CLI、Streamlit UI、常驻服务与三个真实 runner；
- AIGC 冻结适配器和 MobileNetV3 官方基础权重缓存；
- 身份 JSON/NPZ 缓存（启动时仍会校验当前图库/模型 fingerprint）；
- README、requirements、start.bat/start.ps1、环境/数据库状态检查；
- AIGC/身份/底图示例；
- 模型、数据库、依赖、架构、测试、运行状态、修改清单和等价 diff；
- 轻量真实测试证据与 UI 截图；
- 前端 `.venv_frontend`，供本机双击启动优先使用。

## 有意不分发的继承/开发产物

- `packaging/dist`、`packaging/stage`、`packaging/build` 等 v0.1 双搜索旧打包产物；
- `release_v1.0` 旧版本发布目录；
- `runtime/tasks`、`runtime/inbox`、`runtime/validation` 的开发期历史数据；正式发布验证产生的单次真实冒烟任务除外；
- 任意 `__pycache__`、`.pytest_cache`；
- 旧 `build_release.ps1`、`audit_summary.json`、`SHA256SUMS.csv` 和根 `BASELINE_INTEGRITY.md`（它们描述 v0.1 构建，不是 v0.2 发布清单）。

正式版的 `runtime/inbox` 为空；`runtime/tasks` 仅保留正式发布后执行的真实冒烟任务 `case_20260825_152531_95be`，`runtime/validation` 仅保留其摘要 `release_real_smoke.json`。这两项是发布目录本身可运行的审计证据，新任务仍会使用独立 case 目录正常写入。相同任务的紧凑证据另存于只读式 `test_results/release_smoke/`，不依赖运行时目录。

## 外部真实资源

为避免重复数百 MB/GB 数据，正式目录不复制以下本机资源，而由 `config.json` 明确引用：

- 已安装 `C:\Users\lanji\AppData\Local\Programs\DeepfakeDetectionSystem`；
- `C:\Users\lanji\.insightface\models\buffalo_l`；
- `D:\刑侦查竞赛\first_week` 中 A_env、B_env、9 人/45 图身份库和 203 张底图库；
- `D:\python\python.exe` AIGC Python 3.12。

因此这是一份本机正式可运行发布，不是完全离线跨机独立安装器。迁移后必须更新配置并重新执行 `check_environment.py`、`system_status.py` 和真实回归。
