# 修改日志

## v0.4.0 — 2026-08-31

- 从 v0.3.1 独立复制为新开发目录，生成修复前 SHA-256 清单，未覆盖旧版。
- 三模块增加真实中间证据、任务相对 artifact manifest 和更细真实进度事件。
- Streamlit 结果页升级为五页证据驱动界面；新增真实图片对照、候选卡、可交互图表和证据链。
- 相同基准输入的 AIGC、身份、底图分数与 Top-5 排序差值均为 0。
- 正式打包版本、AppId、主程序、启动状态和卸载钩子升级为 v0.4.0。

## v0.2-three-module-official — 2026-08-25

- 从 `face_source_trace_two_search_v0.1_official` 完整复制创建独立 dev，原 official 保持只读不变。
- 审计 AIGC 资料、验收包与已安装程序，确认可验证源码形态为安装目录中的可读模块/打包归档；历史冻结适配器只读加载真实安装版算法和权重。
- 新增 AIGC 冻结适配器、常驻 worker、统一 `detection_result.json` 契约与模型哈希检查。
- 为 AIGC 和 InsightFace 新增独立持久运行服务，模型跨查询复用；保留真实子进程 fallback、超时和进程清理。
- 扩展 orchestrator 为 `deepfake/identity/source` 三分支及 `selected_modules`，实现 skipped、partial_success 和全模块故障隔离。
- 将底图默认超时从 180 秒调整为 600 秒，匹配 203 张图库真实运行时间；算法、权重和排序未改。
- 新增 `system_status.py`，报告 AIGC 资源哈希、身份人物/图片/embedding/index、底图图片/特征/index 和实际运行设备。
- 延续现有 Streamlit UI，增加首页三模块状态、四个操作、进度、错误、三结果页签和无概率融合的证据摘要。
- 增加 `start.bat/start.ps1`、环境检查、示例、真实集成 runner、模型/依赖/数据库/测试/架构文档。
- 将外部 A/B 环境和图库配置改为本机绝对路径，修复 dev 复制到 `releases` 后 `../first_week` 解析错误。
- 完成真实 AIGC、身份、底图边界测试及多案例三模块联合回归；详细数据见测试报告。

## v0.1-two-search-official — 2026-08-20

- 正式集成人脸身份搜索 A 与原始底图搜索 B。
- 建立统一 task、状态、日志、故障隔离与前端结果展示。
- 通过 query_001、query_002、query_003 双搜索真实回归。
