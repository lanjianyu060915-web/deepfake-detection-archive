# UrbanSound8K 原始归属获取计划

530条EnvSDD固定研究集的逐录音作者归属仍缺。官方页面要求保留完整FREESOUNDCREDITS.txt；前1MiB探针和公开Freesound页面未能取得它。原530条<1GiB获取计划及已执行脚本不改写。

本分支只为关闭研究归属缺口：按官方Zenodo固定记录1203745获取6,023,741,708字节UrbanSound8K.tar.gz，先核对官方元数据checksum、完整字节/MD5/SHA。该额外6.1GB在初始30GiB来源资源和100GiB总工作区预算内，当前可用磁盘732GB；不涉及云或付费资源。

不extractall、不发布、不训练原Urban媒体。完整哈希通过后，仅从tar读取白名单原README、FREESOUNDCREDITS.txt及必要metadata CSV，文件须为regular且有限字节；其余媒体只作为压缩来源证据。原压缩包留在忽略cache，不改写EnvSDD530清单或旧64条。原作者归属及每fsID与530条关系单独核验；CC-BY-NC研究边界与比赛授权仍区分。

下载可断点继续，记录请求、原官方元数据、错误、精确字节/哈希及提取成员；任何不一致停止而不宣称许可验收。若官方源不可取，保留缺口，不自动采用第三方许可常量。付费云不启用。
