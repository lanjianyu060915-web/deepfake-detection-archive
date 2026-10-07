# AASIST 源码来源

- 上游：<https://github.com/clovaai/aasist>
- 固定提交：`a04c9863f63d44471dde8a6abcb3b082b07cd1d1`
- 迁移范围：`models/AASIST.py`，保存为 `aasist.py`。
- 修改：文件首部加入 mypy/Ruff/格式化排除注释，保留原模型实现；没有迁移上游训练器或评测器。
- 原始源码、NAVER 授权和完整第三方声明分别保存在 `aasist.py`、
  `AASIST-LICENSE.txt`、`AASIST-NOTICE.txt`。NOTICE 中列出的 t-DCF 评测器没有复制到本包。

类型检查、输入约束与本项目分数方向在相邻 `../aasist.py` adapter 中实现。
CPU 测试核对 AASIST-L 为 85,306 参数，并在本地上游参考存在时逐值核对 eval 输出。
模型结构可训练不代表已复现论文精度；预训练权重、数据许可和真实跨域结果需另外验证。
