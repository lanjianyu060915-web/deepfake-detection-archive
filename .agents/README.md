# 项目级 Agent Skills

本目录保存本项目实际使用的第三方 skills。它们通过固定版本
`npx skills@1.7.1` 以 `--copy` 模式安装，连同仓库根目录的
`skills-lock.json` 一起纳入 Git。协作者克隆仓库后即可读取，无需依赖某台机器的
`~/.agents/skills` 或 `~/.codex/skills`。

## 当前清单

| 来源 | 固定上游 commit | Skills |
| --- | --- | --- |
| `wenmin-wu/ds-skills` | `718b09722905726c428fcc4eab4d470df4842545` | `tabular-group-kfold-leak-prevention`、`cv-phash-duplicate-grouping`、`cv-video-frame-sampling-pipeline`、`cv-frame-prediction-averaging`、`cv-separable-temporal-spectral-cnn`、`cv-mixed-precision-training`、`cv-gradient-accumulation` |
| `openai/skills` | `49f948faa9258a0c61caceaf225e179651397431` | `jupyter-notebook`、`pdf` |
| `huggingface/skills` | `ca0325bb20b2d0a1b2efa893670c4c72f79e707b` | `huggingface-datasets`、`huggingface-papers` |

这些是经过筛选的 11 项项目依赖；不把 ds-skills 的 535 项完整目录复制进仓库。
来源与固定版本以根目录 `skills-lock.json` 为准，第三方许可见本目录
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。这些工具辅助实现，不作为模型实验的前置门。

## 管理命令

```bash
# 查看项目可发现的 skills。
npx --yes skills@1.7.1 list --json

# 按 skills-lock.json 检查并更新项目副本；执行后必须审阅差异。
npx --yes skills@1.7.1 update --project --yes
```

新增时使用明确来源、明确 skill 和复制模式，例如：

```bash
npx --yes skills@1.7.1 add <owner/repository> \
  --skill <skill-name> --agent codex --copy --yes
```

更新或新增后检查上游 commit、`skills-lock.json`、许可、安全扫描结果和 Git
差异，再运行与改动相关的项目验证。第三方 skill 文件保持上游原样；项目特定适配写在
代码或 `docs/` 中。上传、发布、遥测和云任务示例不会因 skill 被收录而自动获得授权。

第三方许可和来源说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
