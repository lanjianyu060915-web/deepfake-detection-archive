# EnvSDD / UrbanSound8K 来源与归属

本轮仅使用 EnvSDD 的 UrbanSound8K 环境音部分开展内部非商业研究。
[EnvSDD 官方项目](https://github.com/apple-yinhan/EnvSDD)及
[固定版本数据卡](https://huggingface.co/datasets/EnvSDD/EnvSDD/blob/b0832726deb55484dbce869c6f09c41f7e9e5fbb/README.md)
登记集合许可 CC BY 4.0；原 UrbanSound8K 的
[官方许可](https://urbansounddataset.weebly.com/)为 CC BY-NC 3.0，继续适用。
本项目未将这些媒体或相关训练权重批准为正式比赛交付资产。

[resources.json](resources.json)保存固定版本、八片训练输入和四片评测输入的官方 LFS SHA-256，
以及原始 metadata/credits 的固定下载地址与哈希。镜像仅作为传输通道，
完整分片下载后按官方哈希核对。

[FREESOUNDCREDITS.txt](FREESOUNDCREDITS.txt)是 SpeechBrain 固定版本保留的
原 UrbanSound8K 逐录音署名副本。导入时由 metadata 的 fsID 关联上传者，
输出逐样本 provenance；原录音、其切片及生成版本不跨训练/评测划分。
UrbanSound8K 引用：Justin Salamon、Christopher Jacoby、Juan Pablo Bello，
*A Dataset and Taxonomy for Urban Sound Research*，ACM Multimedia 2014。

所选 EnvSDD train/validation 部分包含 AudioGen、AudioLDM1、AudioLDM2，
其中 AudioLDM1/2 属同一方法家族。四种生成方法为三个 TTA 和 AudioLDM1 ATA。
训练不使用 TangoFlux、AudioLCM，亦未继承上游 train/validation 划分。
另取 `remain` 的第25、26、38、39片，只把其中 AudioLCM/TangoFlux 的 UrbanSound8K
生成版本按原文件名关联到既有 dev/holdout。二者仅用于未见生成器评测，不进入训练。
此处“未见”指生成器未参与训练；评测音频仍来自同一城市环境音语料，不能代表跨来源能力。
