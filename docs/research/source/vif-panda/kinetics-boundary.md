# Kinetics 媒体边界

本项目不将当前 ViF-CoT-4K `source_videos.zip` 中 `kinetics/` 媒体用于训练。

DeepMind 的 Kinetics I3D 仓库说明，原始 Kinetics release 只包含
“YouTube IDs and the start and end times of the clips”。因此，这不是一个对视频字节的
通用再授权。CVDF 后续托管媒体的下载说明没有提供替代的视频许可文本。

YouTube Terms of Service（2025-03-17）限制对 Content 的下载、复制和分发，除非服务
明确允许、取得 YouTube 与适用权利人的书面许可，或适用法律允许。此处仅归档该不超过
25 词的边界摘要与来源，不复制长条款全文。

- DeepMind source: <https://github.com/google-deepmind/kinetics-i3d#sample-data-and-preprocessing>
- CVDF downloader: <https://github.com/cvdfoundation/kinetics-dataset#download-videos>
- YouTube Terms version/date: <https://uk.youtube.com/t/terms>

保留 Kinetics ID 与配对 metadata 可以帮助审计，但不能使媒体训练获得授权。竞赛外部数据
规则仍为 `unconfirmed`。
