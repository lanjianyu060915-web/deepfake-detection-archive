# D-v2 鉴伪独立验收体系

本目录是“考卷规范与阅卷系统”，不是本轮真实模型考试结果。当前鉴伪 baseline、模型、阈值、`0.62 / 0.38` 融合和预处理均保持只读；本轮没有对 Core Blind 运行真实鉴伪。

## 数据分层

- `Core_Blind`：首次正式主成绩，只能接收具有证据真值、未参与训练/调参/既往观察且通过独立性签署的原图。
- `Perturbation`：由可信原图确定性生成，标签继承，按 `parent_sample_id` 聚合；不得伪装成独立样本。
- `Development_or_Exposed`：训练、验证、既往测试、开发夹具和已观察样本，不进入首次 Core Blind。

候选目录名、文件名、肉眼判断和模型输出都不是合法真值证据。证据不足统一写 `UNKNOWN`，而正式 ground truth 只允许精确的 `REAL` 或 `FAKE`。

## 推荐流程

```powershell
D:\python_3.14\python.exe tools\audit_candidates.py `
  --workspace D:\刑侦查竞赛 `
  --inventory candidate_inventory\CANDIDATE_INVENTORY.csv `
  --overlap candidate_inventory\OVERLAP_CHECK.csv `
  --summary candidate_inventory\AUDIT_STATS.json

D:\python_3.14\python.exe tools\build_manifest.py `
  --curation-csv dataset\manifests\human_curated_ground_truth.csv `
  --dataset-root dataset `
  --output dataset\manifests\deepfake_ground_truth.csv

D:\python_3.14\python.exe tools\validate_dataset.py `
  --manifest dataset\manifests\deepfake_ground_truth.csv `
  --dataset-root dataset `
  --report-json reports\DATASET_VALIDATION.json `
  --report-md reports\DATASET_VALIDATION_REPORT.md
```

当前模板为空是有意设计：候选数据尚不满足 Core Blind 独立性，工具不会为了凑数自动贴标签或复制 200 张图。

正式评测器的标签映射严格、区分大小写：adapter 的 `real` 映射到 `REAL`，`deepfake` 映射到 `FAKE`；其他任何值均是系统失败并按固定分母计错。真实运行前须按 `protocols/D_V2_FORMAL_BLIND_TEST_PROTOCOL.md` 完成签署与冻结。
