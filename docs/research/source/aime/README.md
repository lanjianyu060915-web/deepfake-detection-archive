# AIME 音乐研究来源

使用 [disco-eth/AIME](https://huggingface.co/datasets/disco-eth/AIME)，固定 revision
`b84d4be5eda830b6eb714998569dba73530f2601`。上游共有500条真实 MTG-Jamendo
音乐和12个模型的6,000条生成音乐；本轮选取90个原始 Parquet 分片，
共9,087,224,372字节。固定路径、大小及官方 LFS SHA-256 在
[resources.json](resources.json)，不是整个62.28GB数据集。

输入字段仅有 `id/model/description/audio`。本轮以真实样本的三标签描述筛选生成音乐，
按忽略顺序、大小写的标签组合保守分组，再连接已确认的处理后精确重复。
这种语义匹配不是原曲波形派生关系。上游没有 Jamendo 原曲 ID、artist 或 album 映射，
不能声称已按艺术家隔离；AIME 行 ID 仅标识发布样本。

训练生成路径为 MusicGen Small/Medium、AudioLDM 2 Large、Riffusion 和 Mustango；
剩余选中生成器只作开发诊断。两个 MusicGen 尺寸属于同一家族；MusicGen Large
即使未训练，也不能当成未知家族。真实和生成音频统一下混、16kHz、居中4秒、
峰值0.95及PCM16处理；原始长度、采样率、声道仅记录，不输入模型。

## 许可与用途

[AIME 数据卡](https://huggingface.co/datasets/disco-eth/AIME/blob/b84d4be5eda830b6eb714998569dba73530f2601/README.md)
将生成音乐声明为 CC BY 4.0，标签来自 MTG-Jamendo、为 CC BY-NC-SA 4.0。
真实音乐保留原曲各自的 Creative Commons 许可，不能统一改成 CC BY。
[MTG 官方用途说明](https://mtg.github.io/mtg-jamendo-dataset/#license)
允许非商业研究与学术使用，本轮训练仅在该范围内。

[固定原始逐曲许可清单](https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/cafd8e20c265ed84f1e61f1c875327971f43a62f/audio_licenses.txt)
包含 BY、BY-SA、BY-ND、BY-NC、BY-NC-SA、BY-NC-ND 等不同条款和版本。
AIME未提供原曲映射，逐曲许可及署名尚不能解析；以
`LicenseRef-MTG-CC-per-track` 保留此事实。内部研究授权不等于允许正式比赛交付或公开分发
处理后的真实音乐，当前 `competition_approved=false`。

来源论文：[Benchmarking Music Generation Models and Metrics via Human Preference Studies](https://arxiv.org/abs/2506.19085)。
本轮不是该论文的人类偏好评测，也不是 AIME 官方鉴伪基准。
