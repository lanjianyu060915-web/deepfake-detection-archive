# AI-OpenBMAT 混合音频研究来源

使用 [AI-OpenBMAT Zenodo 记录 20031232](https://zenodo.org/records/20031232)，
完整 ZIP 为7,312,205,691字节，官方MD5为
`436f1bcc46e9455fc4f732205b8dcb99`。
发布包有1,647组真实/AI配对，共3,294条60秒音频；排除 `__MACOSX` 元数据镜像。
论文方法与字段见[官方论文](https://arxiv.org/html/2602.06823v1#S2)及
[固定版本JSON样例](https://github.com/DaveLoay/AI-OpenBMAT/blob/a9dfc562f0e5f657029a14de09ec01238292dec6/assets/examples/set-01-spain-documentary/gen_00023_ai.json)。

本轮按 `sha256("42:" + 五位pair_id)` 排序固定选取128对，ID见
[资源清单](resources.json)。通过HTTP Range提取对应WAV与JSON，检查各ZIP条目的
大小与CRC32，并在本地任务资源表记录SHA-256；没有下载或校验完整ZIP，不能声称
整包MD5已验证。原始文件保留上游 `human/gen_XXXXX.*` 和 `ai/gen_XXXXX_ai.*` 命名。

## 片段与用途

每对分别选一个最长的共同纯音乐区间和一个最长的共同音乐/语音混合区间。
依据两份JSON中的实际 `segments.duration` 重建时间轴，每次连接扣除
`config.crossfade_ms`；在共同区间两端各留0.1秒，再中心截取4秒。
共同区间不足4.2秒的桶不选取；无音乐区间不承袭AI文件的fake标签。
真假使用同一起点、时长与处理：单声道16kHz、`resample_poly`、峰值0.95、PCM16。
缺失、损坏及越界音频逐样本记账，不从覆盖率分母移除。

全部样本仅作外部探索dev，不加入本轮训练或holdout。
128对最终得到246段：混合152段（76real/76fake）、纯音乐94段（47real/47fake），
0准备失败；分别52对与81对不满足对应桶的区间长度条件。
两桶不使用完全相同的pair集合，不能仅凭两桶总体EER差异归因于混合语音。
配对ID绑定两个版本；provenance保留音乐、广播原片及参考结构关系和截片位置。
多条成品会复用同一音乐或广播素材，pair数不等于独立原片数；若后续用于训练，
须按这些关系连接分组，不能直接随机划分成品或4秒片段。
来源库为BAF中的Epidemic Sound，与AIME的MTG-Jamendo不同，但曲目/artist重叠未核实。

**标签来自上游。** Suno v3.5使用Extend续作，官方未说明发布音乐是否去除了真人提示前缀，
JSON也没有生成内容起点。按含音乐区间截片仍不能保证每条AI侧4秒窗口均含生成音乐，
因此这些分数不能当成确定生成片段的金标准；不自拟跳过前若干秒来声称解决。

## 准备与许可

共享原始子集位于
`/var/tmp/deepfake-audio-regularize-20261012/upstream/ai-openbmat/raw/`。
对资源清单对应文件完成提取后，使用新输出目录：

```bash
scripts/devctl run python tools/prepare_ai_openbmat.py \
  --raw-dir /path/to/raw --output /path/to/new-ai-openbmat
```

输出包含纯音乐与混合两份dev清单、来源、provenance和统一处理后的媒体。
本机可直接读取同任务根的 `ai-openbmat/`，模型结果见
[时频遮挡与广播内容实验](../../../experiments/audio-ast-regularization-20261012/README.md)。
本机HTTP Range条目表及每个原文件的本地SHA在同任务根的
`upstream/ai-openbmat/resources.json`。

AI-OpenBMAT发布标为CC BY 4.0；但[BAF原始说明](https://github.com/guillemcortes/baf-dataset/blob/main/README.md)
限定非商业研究，并禁止音乐生成。上游未说明这些底层素材权利如何衔接，
因此仅登记研究用途，`license_status=unresolved`、`competition_approved=false`，
不进入正式交付或公开重分发。
