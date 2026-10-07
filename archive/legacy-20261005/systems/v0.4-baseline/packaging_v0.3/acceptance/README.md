# 安装后验收

`phase1_install_start.ps1` 把 v0.4.0 安装到本次运行目录下的
`中文 空格安装/应用`，隔离用户数据目录，校验资源、注册表、快捷方式、版本化
状态文件、端口和重复启动，然后保持系统运行供浏览器验收。

浏览器验收必须真实运行 AIGC、身份、底图和一键综合分析，并验证 JSON/TXT/HTML
报告。完成四次真实任务后运行：

```powershell
pwsh -File .\collect_business_evidence.ps1 -RunRoot <PHASE1_READY 输出目录>
```

采证脚本从隔离的 `DATA_ROOT/tasks` 按 `selected_modules` 精确选择四类最新任务；也可
通过 `-DeepfakeTaskId`、`-IdentityTaskId`、`-SourceTaskId` 和
`-ComprehensiveTaskId` 明确指定。它严格复核任务状态、三模块结果、统一结果、
进度终态和三格式报告，并要求底图扫描完成到实际解码总数、几何复核完成到
shortlist 总数。缺失、歧义、Mock 标记或跨文件不一致都会报错，不会生成通过证据。
默认输出为 `<RunRoot>/business_evidence.json`，直接作为 phase2 的
`-BusinessEvidencePath`。禁止复制示例中的占位值冒充通过。

`phase2_restart_uninstall.ps1` 读取业务证据，停止系统、再次启动并检查健康状态，
再次停止、静默卸载，验证注册表和快捷方式消失、用户报告保留；如 phase1 指定
受保护旧安装目录，还会比较卸载前后逐文件 SHA-256 清单。

验收目录及安装器路径均为参数，不依赖固定盘符或用户名。脚本不会递归删除
验收根目录，也不会清理保留的用户任务证据。
