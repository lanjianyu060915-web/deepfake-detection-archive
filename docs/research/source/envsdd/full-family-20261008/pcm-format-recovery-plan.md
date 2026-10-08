# 实际 PCM 位深差异：保留源字节的恢复计划

同一固定revision/清单续下载时，已有64条明确失败源于sampwidth=3的有效24位PCM。原脚本、原失败账本保留；不把有效来源格式差异当作数据坏损。

在新 acquire_pcm.py 支持WAV原生unsigned8/signed16/24/32位PCM，保持16k mono/64000帧/4s约束和每body256088字节界限。完整PCM字节数须等于frames×samplewidth，额外帧/截断/非PCM仍拒绝。纯整数正确符号扩展到float64诊断，不转码或改写原来源字节。增加16/24/32位符号边界与截断负例，独立verify_pcm.py使用libsndfile float64解码交叉核验，继续重算原清单、刷新元数据、原字节哈希和无缺样/重复。

仅原运行结束后从同一固定清单和已有验证账本恢复，保留新运行与脚本hash。最终报告按label/generator/fsID/event审计位深，原real与生成的格式不对称不得忽略；本分支不因此批准模型训练、比赛许可或完整来源归属。旧64条与原530选择不改写。
