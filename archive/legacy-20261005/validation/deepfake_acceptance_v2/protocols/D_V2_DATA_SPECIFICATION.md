# D-v2 数据规范

## 1. 目标与分层

核心独立测试集目标为 200 张原图：`REAL=100`、`FAKE=100`。目标不是配额授权；证据不足的样本一律为 `UNKNOWN`，不得为了平衡强行定类。

```text
D-v2/
├── Core_Blind/                 # 第一次正式主成绩
│   ├── REAL/
│   └── FAKE/
├── Perturbation/               # 相关衍生鲁棒性诊断
└── Development_or_Exposed/     # 不计首次独立成绩
```

## 2. 真值准入

合法证据包括明确 ground truth、CSV/JSON 标签、README 的确定性说明、生成脚本/日志和成员签署映射。文件名、目录名、肉眼感觉和模型预测均不构成真值。

- REAL：须有原始采集/来源链路以及未做换脸或生成处理的证据；“原图”“Open Images 来源”本身不足。
- FAKE：须有明确生成记录或确定性真值映射；尽量保留生成程序/版本/时间、source_face、target_face 和原始生成文件。
- UNKNOWN：只能留在候选清单，绝不进入正式 ground truth。

正式 CSV 固定字段和顺序：

```text
sample_id,file_path,label,source,generation_method,source_face_id,target_face_id,is_original,parent_sample_id,perturbation_type,perturbation_level,width,height,sha256,notes
```

`label` 只允许区分大小写的 `REAL` / `FAKE`；`is_original` 只允许 `true` / `false`。路径必须相对 dataset root 且不得越界。

## 3. 独立性准入

Core Blind 样本不得参与当前或上游训练、验证、阈值选择、融合调参、既有正式指标或开发人员预测观察。必须核对 SHA-256、文件名、路径及可行时的感知哈希，并填写独立性声明。训练数据完整清单不可得时，状态必须是 `UNKNOWN`，不可宣称“无泄漏”。

同一内容的多路径只按唯一 SHA-256 计一次。同一人物/视频的相邻帧不得跨训练与测试划分。Core 在测试前由独立保管人封存，模型开发人员不可预览。

## 4. 覆盖建议

REAL 尽量覆盖不同人物、场景、设备、分辨率、光照、姿态和自然压缩。FAKE 建议至少 3 种有记录的生成方法，同时避免 source/target 人物泄漏。每种方法和场景都记录实际数量；不知道的字段写 `unknown`，不得推测。

## 5. 扰动规范

只从已准入的原图生成：JPEG Q90/Q70/Q50、Scale 0.75/0.5、Gaussian blur 1/2、Brightness -15%/+15%。衍生样本继承父标签，文件名含 parent/type/level，记录实际尺寸、SHA-256 和 `parent_sample_id`。

原图与 JPEG/Scale/Blur/Brightness 分组报告；含衍生样本的 micro 指标只能称为相关诊断，并必须给 parent 聚合，不能包装成独立样本总体准确率。

## 6. 冻结门槛

只有在 100/100 均具备证据、独立性签署完成、无重复、validator 为 PASS、代码/模型/evaluator/manifest 全部冻结且尚未运行模型时，才能写 `READY_FOR_FORMAL_BLIND_TEST`。否则工程可以通过，但数据状态为 `ENGINEERING_PASS_DATA_PENDING`。
