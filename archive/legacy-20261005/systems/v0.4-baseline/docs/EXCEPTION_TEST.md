# 异常与故障隔离测试

## query 不存在

```text
受控错误：查询图片不存在或不是文件：...\不存在的查询图.jpg
EXIT_CODE=2
```

结论：非零退出、错误明确、无未捕获 Traceback、未创建 task，PASS。

## A 失败但 B 继续

`tests/test_orchestrator_contract.py` 使用 `failing_identity_runner.py` 与 `successful_source_runner.py`，只验证编排控制流，不修改或替换正式 A/B：

- A result/status = error。
- B runner stdout = `SOURCE_FIXTURE_RAN`。
- B result/status = success。
- unified/task status = `partial_success`。
- A/B 日志均存在。

```text
3 passed in 0.19s
```

结论：故障隔离 PASS。

