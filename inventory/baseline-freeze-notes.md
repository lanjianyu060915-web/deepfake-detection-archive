# 换脸寻源 v0.4.0 源系统冻结包

冻结时间：2026-09-20 09:20–09:29（Asia/Shanghai）
冻结人：Codex（受兰建渝委托执行）

---

## 这个包是什么

把 **v0.4.0 冻结版源系统**在"未做任何升级改动"的状态下完整打包留档，
用于防止后续在 D 盘开发升级时误删、误改或磁盘故障导致原始版本丢失。

**源目录（只读，未做任何修改）：**
`D:\刑侦查竞赛\face_source_trace_three_module_v0.4.0_visualization_dev`

---

## 目录内容

| 项目 | 说明 | 大小 |
| --- | --- | --- |
| `源系统_v0.4.0_完整冻结_20260920.zip` | 源系统完整归档（41,464 个文件） | 5,484 MB |
| `SHA256SUMS_源系统_v0.4.0.csv` | 逐文件 SHA-256 清单（41,464 行） | 6.3 MB |
| `_archive_sha256.txt` | 归档本身的 SHA-256 | — |
| `发布交付包\` | v0.4.0 官方发布交付包（含 Setup.exe、截图、技术文档） | 约 908 MB |
| `换脸寻源系统_v0.4.0_安装器.exe` | 从 `D:\系统3.0` 复制的安装器 | 15.9 MB |
| `_freeze_progress.json` | 冻结过程的机器可读记录 | — |

### 归档内顶层目录

解包后得到 `face_source_trace_three_module_v0.4.0_visualization_dev\`，
与源目录结构逐级一致，含源码、前端、测试、打包资源、运行日志与测试结果。

---

## 完整性凭证

| 校验项 | 结果 |
| --- | --- |
| 源文件数 | 41,464 |
| 源总大小 | 6,525,350,835 字节（6.53 GB） |
| 归档条目数 | 41,464 |
| 归档大小 | 5,484 MB（压缩率 0.881） |
| **归档 CRC 全量校验** | **全部通过** |
| 清单条目数 | 41,464 |
| 清单有而归档缺 | 0 |
| 归档有而清单缺 | 0 |

| 文件 | SHA-256 |
| --- | --- |
| `源系统_v0.4.0_完整冻结_20260920.zip` | `5E8B6D04EB496C425B15FDCA938BFF47A9581F5404E7877D73507F96256675E3` |
| `SHA256SUMS_源系统_v0.4.0.csv` | `982EAF4025571D04FD2C72640D1970B30132CC0C4E99377A046EE57A30228E9B` |

---

## 怎么校验

```powershell
# 1. 归档本身是否完好（先看这一步）
(Get-FileHash '.\源系统_v0.4.0_完整冻结_20260920.zip' -Algorithm SHA256).Hash
# 应等于 5E8B6D04EB496C425B15FDCA938BFF47A9581F5404E7877D73507F96256675E3

# 2. 解包后逐文件核对（可选，耗时较长）
python -c "
import hashlib, csv, sys
from pathlib import Path
root = Path(r'解包后的目录')
bad = 0
for row in csv.DictReader(open('SHA256SUMS_源系统_v0.4.0.csv', encoding='utf-8-sig')):
    p = root / row['relative_path']
    if not p.is_file():
        print('缺失', row['relative_path']); bad += 1; continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != row['sha256']:
        print('不一致', row['relative_path']); bad += 1
print('核验完成，异常项 =', bad)
"
```

---

## 怎么还原

```powershell
# 解包到任意目录即可得到完整源码工程
Expand-Archive -LiteralPath '.\源系统_v0.4.0_完整冻结_20260920.zip' `
               -DestinationPath 'D:\还原\'
# 还原后目录：D:\还原\face_source_trace_three_module_v0.4.0_visualization_dev\
```

想恢复成"能直接跑的系统"，优先用 `发布交付包\换脸寻源系统_v0.4.0_Setup.exe`（安装版），
源码工程用于重建与追溯。

---

## 注意事项

1. **本包是升级前的纯净态。** 后续在 D 盘做的所有二代检测器改动（`deepfake_detection_v2`、
   `packaging_v0.4` 等）都在 `D:\刑科技项目升级2.0` 内，**不在本包中**，这是刻意隔离的。
2. 本包位于 C 盘，与 D 盘的开发目录物理分离；但仍在同一台机器上，
   正式提交前建议再复制一份到移动硬盘或网盘。
3. 归档为单文件 ZIP64，含 41,464 个条目；资源管理器解压较慢，建议用 `Expand-Archive` 或 7-Zip。
4. `packaging_v0.3\work\` 等构建中间产物约占一半体积，属于可再生产物，
   若日后需要精简版可以排除，但**本冻结包保留了全部内容**以做到逐字节可还原。
