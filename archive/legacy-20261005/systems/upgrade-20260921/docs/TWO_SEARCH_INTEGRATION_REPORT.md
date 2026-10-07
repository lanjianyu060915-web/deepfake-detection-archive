# 双搜索正式集成报告

| Case | 身份状态 | 身份Rank1 | 底图状态 | 底图Rank1 | 总耗时 | Task隔离 |
|---|---|---|---|---|---:|---|
| query_001 | success | person_001 | success | 122.jpg | 96.508606 秒 | PASS |
| query_002 | success | person_002 | success | 107.jpg | 42.238987 秒 | PASS |
| query_003 | success | person_003 | success | 019.jpg | 39.507439 秒 | PASS |

## 验收结论

- 三个真实案例：`3/3` 成功。
- A Rank-1 回归：三例均与第1.5轮首次正式结果一致，PASS。
- B Rank-1 回归：三例均与第1.5轮首次正式结果一致，PASS。
- 统一入口：三例全部由 `launcher.py → orchestrator.py → A_env/B_env subprocess` 运行。
- 人工运行 A/B：否。
- 人工复制 JSON 或移动匹配图：否。
- Task 隔离：三个 task_id、目录、结果路径和 15 张匹配图路径全部唯一，PASS。
- 必需归档：每例 query、A/B JSON、统一结果、task 状态、四类 stdout/stderr 日志均存在。
- 候选数量：每例 A/B 各 5 个；每例 5 张 B 匹配图均存在并位于本 task。

## 冻结完整性

集成前与全部运行后均复核：原 A 15/15、原 B 22/22、buffalo_l 5/5 为 MATCH；产品 A 运行子集 10/10、B 运行子集 11/11 为 MATCH。A/B 算法和模型未修改。

默认产品配置使用原始 45 张身份参考图与 203 张底图库；第1.5轮正式 profile 使用从同一来源物化的 44/200 选择集。三例在产品默认库下得到与冻结正式结果相同的 Rank-1 与分数，本报告不声称两个候选集合完全相同，也不重算总体准确率。

## 前端

- `run_app.bat` 启动成功。
- `GET /_stcore/health`：HTTP 200，响应 `ok`。
- `GET /`：HTTP 200，返回 Streamlit 页面。
- 默认模式：Live Analysis。
- 真实页面上传 `query_001` 后，新建 `case_20260820_162420_59ac` 并完成 A/B。
- 页面异常 0、错误 0；身份 Top-5、5 张参考图、底图 Top-5、5 张候选图、5 张匹配可视化和模块/总耗时均可加载。
- 验证完成后已停止服务，端口不保留监听。

截图不是硬门槛；本轮采用 HTTP、真实 Live Analysis 页面驱动、session state、结果组件和资源存在性作为可复核证据，未增加浏览器截图依赖。

## 异常测试

1. query 不存在：launcher 返回 2，输出明确“受控错误”，没有未捕获 Traceback，也没有创建 task，PASS。
2. A 失败但 B 继续：使用独立 fixture 模拟 A runner 失败与 B runner 成功；B stdout 证明已运行，task=`partial_success`，PASS。
3. 工程测试：`3 passed in 0.19s`。

Streamlit AppTest 在所有页面断言通过、进程退出码为 0 之后，Python 3.14 测试框架的 weakref 清理曾输出临时目录 ACL 提示；相关临时目录已按精确路径清理，未进入产品、release 或复核包。该提示未出现在产品页面、服务或 task 日志中。

## Git

`D:\刑侦查竞赛` 不是 Git 仓库。

```text
Git tag not created: repository not present.
```

## 最终判定

```text
ROUND = PASS
```

