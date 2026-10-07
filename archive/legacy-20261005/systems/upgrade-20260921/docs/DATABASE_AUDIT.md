# 身份库与底图库审计

审计日期：2026-08-25  
审计方式：只读文件盘点、图片解码、缓存 JSON/NPZ 结构与哈希检查；未重建 index。

## 身份数据库

- 路径：`D:\刑侦查竞赛\first_week\data\identity_library`
- 人物数：9
- 图片数：45 JPG
- 总大小：159,485,673 字节
- 解码检查：45/45 可解码
- 模型：InsightFace `buffalo_l` 五个冻结 ONNX；总计 341,268,726 字节，代码内固定大小/SHA 与实文件全部匹配
- 检测尺寸：1280×1280
- embedding：512 维 float32，L2 归一化
- 相似度：查询与同一人物各参考图做余弦相似度（归一化点积），人物得分取参考图均值；同分以 identity_id 排序；Top-K 默认 5

### 持久缓存/index

身份 index 不是 FAISS/SQLite，而是安全 JSON 元数据 + NPZ 数组：

- `runtime\cache\identity\identity_index_0f638bd4cc311b73e202db6c.json`
- `runtime\cache\identity\identity_index_0f638bd4cc311b73e202db6c.npz`
- 当前缓存形状：`(43, 512)`，float32、C 顺序、全为有限值，覆盖 9 人
- NPZ SHA-256 与 JSON 的 `data_sha256` 一致：`75bd239089ab5f65d6ac0b4ced3d71724596562f2031b387e9c42409107693b6`
- 当前 45 张图库文件的路径/大小/mtime/SHA、buffalo_l 冻结模型 fingerprint、1280×1280 检测尺寸和 embedding 策略重新计算出的 fingerprint 与缓存一致，因此可命中缓存
- 两张未进入当前 embedding：`person_001/06.jpg`、`person_002/01.jpg`。两图可以解码；缓存没有记录跳过原因，不能臆测为无人脸或损坏。

UI 状态口径：库和五个 ONNX 的大小/SHA 全部通过且人物/图片非空时显示“可用”；缓存 fingerprint、NPZ SHA、shape、dtype 和有限值全部通过时另显示“index 可用”。库为空、缺失、模型损坏或缓存过期时立即给出不可用/需重建状态，不用假绿灯。

## 原始底图库

- 路径：`D:\刑侦查竞赛\first_week\data\source_library`
- 图片数：203（201 JPG + 2 PNG）
- 总大小：112,099,984 字节
- 解码检查：203/203 可解码

底图模块没有深度学习 embedding、FAISS、SQLite 或持久特征 index。每次检索真实扫描全部图片：

1. Haar 最大人脸 + 25% 扩框生成遮罩；无脸则用全图；
2. 全库计算 HSV、灰度、边缘与外围特征；
3. Top-5 时取全局粗排前 20；
4. SIFT + BF L2 KNN + Lowe ratio 0.75 + RANSAC homography；
5. `final_score = 0.45 * global_score + 0.55 * local_score`。

UI 状态口径：图库存在且图片非空时显示“可用（实时全库扫描，无持久 index）”。不得伪报已有 embedding/index。图库为空或缺失时立即显示不可用并禁用底图按钮；不进入无限 loading。

## 当前性能风险

最近官方真实任务中，身份约 19.4 秒、底图约 148.7 秒、总计约 168.1 秒。底图源码默认 180 秒余量过小，本集成将默认 timeout 调整为 600 秒；该变化只影响任务保护，不改变检索算法或得分。
