# 前端真实验收记录

日期：2026-08-25  
服务：`http://127.0.0.1:8767`  
工具：Codex 应用内浏览器，真实 Streamlit 服务。

## 验收结果

- 页面标题为“AI换脸涉案图像智能鉴伪与原始素材溯源系统”。
- 首页状态不是 mock：AIGC 可用；身份 9 人/45 图/43 embedding/index ready；底图 203 图/index not_persistent；实际设备 CPU。
- 侧栏报告 AIGC/身份常驻服务 ready，模型可跨查询复用。
- 未上传时四按钮禁用；上传受支持图片后全部启用。
- 一键综合分析显示模型加载、图像检测、人脸/身份特征、身份库搜索、底图特征和底图库搜索进度。
- `case_20260825_150652_5edf` 三模块均 success，总耗时 53.256 秒。
- 证据汇总并列呈现 AIGC、身份 Top1、底图 Top1，并明确禁止跨模块融合概率。
- AIGC、身份 Top-5、底图候选/可视化三个页签均能读取真实 task，并通过结果 schema 与本地资源路径安全检查。
- 测试结束后临时 Streamlit 服务、AIGC worker 和 identity worker 均已退出；未触碰占用 8766 的现有双搜索程序。

## 截图

- `screenshots/three_module_home_ready.png`
- `screenshots/three_module_live_progress.png`
- `screenshots/three_module_live_result_summary.png`
- `screenshots/three_module_identity_results.png`
- `screenshots/three_module_source_results.png`

同一批截图的轻量副本位于 `test_results/ui/`，用于复核包。

