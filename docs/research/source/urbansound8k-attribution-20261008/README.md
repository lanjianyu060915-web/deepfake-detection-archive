# UrbanSound8K 原录音归属恢复

官方固定记录1203745的文件大小6,023,741,708字节、MD5 `9aa69802bbf37fb986f71ec1483a196e` 已从官方API核对；原始记录及body SHA保存。完整包正在忽略cache获取，观察时进度见[resume-checkpoint.json](resume-checkpoint.json)，这不是完整hash验证。

固定530条EnvSDD清单使用40个fsID。`fetch_recording_attributions.py`读取原站HEAD重定向，40/40规范URL与原ID匹配，登记27个用户名；逐行原站状态/Location/日期及失败尝试保存。用户名是规范上传者URL中的标识，不冒称真实姓名或录音内容中的人物；它不等于逐录音许可原文或完整官方credits文件。未经准入仍不训练/正式比赛使用。

归属条件由[Urban官方页面](https://urbansounddataset.weebly.com/)确认，原始文件后续需要保留作者/出处及研究边界，不能根据另一个工具的常量放宽上游条件。完整压缩来源只为读取credits/README/必要元数据，不extractall或使用原Urban媒体。

## 后续恢复

1. 先查真实下载进程与文件大小；工具handle失效不证明进程停止，不重复下载或覆盖在写文件。
2. 若实际进程停止，可用同一官方API及curl断点继续，精确命令如下；保留现有.partial文件。
3. 文件达到精确大小后执行 `finalize_attribution.py`，先完整MD5/SHA，再只读有界白名单文本。未完成时守卫明确拒绝且不创建assets目录，本轮负例已通过。
4. 原credits成功获取后逐40fsID核验归属，与本轮原站记录交叉核对；原许可、生成器/派生许可、研究训练准入与正式比赛资源许可分别据证据决定。

```bash
curl --fail --location --continue-at - --retry 3 --retry-delay 2 --connect-timeout 20 --speed-time 60 --speed-limit 1024 --proto '=https' --output cache/downloads/UrbanSound8K.tar.gz.part 'https://zenodo.org/api/records/1203745/files/UrbanSound8K.tar.gz/content'
.venv/bin/python -B docs/research/source/urbansound8k-attribution-20261008/finalize_attribution.py
```

下载stdout/stderr为 `outputs/experiment-planning/urban-attribution-download.*`。归属获取两个脚本Ruff/format/strict mypy通过；HTTP重定向1正例4负例通过。尚未对完整档案提取做实际验收，不以脚本准备或完整源码清单替代这一证据。
