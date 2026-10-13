# 三域通用音频联合训练

本轮为 #41 的训练实现与配置，尚未完成正式候选冻结和跨来源验收。
复用 AST 12层、4秒输入，按语音/音乐/环境音以及真假采样；dev 用分域宏平均 EER
选模，各域上限分别为语音10%、环境音5%、音乐10%。无合格 epoch 时保留最低 macro
的研究诊断权重，明确 `selection_gate_passed=false`，不把它视为合格交付候选。
`best_dev_eer` 表示所选 epoch 的 pooled EER，选模指标另记
`best_selection_metric` / `best_selection_value`。

注册表共68,472条：train40,167 / dev28,017 / holdout288；holdout只保留划分元数据，
准备和训练不读取其媒体。来源及运行产物在团队共享目录
`/var/tmp/lanjianyu-issue41-20261013/`，代码和配置更新不修改正在运行的工作副本。
音乐仅有prompt分组，原曲/artist独立性未知；music-aime96和speech-asv200均来自当前dev，
不能当作独立泛化结果。AIME逐曲条款与正式赛用适用性仍未解决，候选保持研究用途。

现有2epoch profile的选中轮次：环境音3.3362%、音乐0.8333%、语音0.9027%，
macro1.6907%、pooled1.4007%，各域均低于配置上限。旧工作副本曾把macro记为
`best_dev_eer`，本PR已修正新运行的字段；上述数字按实际dev报告区分，未改写旧产物。
20epoch实验尚无完成报告，不据目录中的旧`running`字段宣称进程仍活跃或模型已验收。

当前注册表每域恰有一个source，因此等权配置为各域约1/3；
`joint-speech-40pct-seed42.json`的0.4/0.3/0.3将语音提高到40%，不是降低。
保留其中原运行目录名`joint-speech-down-seed42`以追溯已经启动的实验，参数不改写。
待完整训练完成，选定一个候选后再验收独立holdout；既有holdout评分失败不能计为完成。
