# 团队开发环境

本页描述现有RTX4090服务器的显式兼容入口。通用安装与运行见[开发指南](development.md#环境)，
发布包使用自己的`run.sh`。以下服务器路径不作为公共安装或模型发现的默认前提。

## 已有共享cu132环境

环境为`/opt/venvs/deepfake-cu132`，只由有写权限的维护者同步；普通任务不重新安装依赖：

```bash
scripts/devctl run python -m deepfake_detection_archive --help
scripts/devctl test tests/test_unified_predict.py
scripts/devctl check
# 仅需要更新依赖时，由维护者执行；保留共享同步锁
scripts/devctl sync
```

`devctl`默认保留团队cu132环境，`UV_PROJECT_ENVIRONMENT`可显式覆盖目录。
它优先当前checkout的`src/`，不往共享环境安装某个工作副本。
不再强制覆盖`CUDA_HOME`、`CUDACXX`或`UV_PYTHON_INSTALL_DIR`；编译研究扩展需要的工具链
由对应任务显式设置。`doctor`报告实际Python、torch构建、CUDA/GPU与媒体工具，
无GPU时可以诊断CPU，不要求所有开发者都是CUDA13.2或RTX4090。

## 独立环境的可选快捷入口

```bash
DEEPFAKE_BACKEND=cpu scripts/devctl sync
DEEPFAKE_BACKEND=cpu scripts/devctl run python -m deepfake_detection_archive --help
DEEPFAKE_BACKEND=cpu scripts/devctl check
# cu126使用另一个目录，不把CPU或cu126同步到共享cu132环境
DEEPFAKE_BACKEND=cu126 scripts/devctl sync
```

默认分别使用该checkout的`.venv-cpu`、`.venv-cu126`，首次由uv创建；
指定`UV_PROJECT_ENVIRONMENT`时使用该目录。
`DEEPFAKE_BACKEND`决定sync依赖组合，显式目录不会自动改变后端，避免将共享环境切成CPU。
三种后端使用不同环境；不把本机旧CUDA环境作为本任务CPU环境。

## 可选direnv

仓库`.envrc`不注入服务器路径、不安装依赖。需要自动激活时，在Git忽略的`.envrc.local`
明确选择已有环境，审阅后运行`direnv allow .`：

```bash
# 服务器示例；原电脑可替换成自己的独立环境路径
export UV_PROJECT_ENVIRONMENT=/opt/venvs/deepfake-cu132
export DEEPFAKE_BACKEND=cu132
```

`.envrc`会激活已指定且存在的环境，并让当前源码优先；没有本地配置时不选择共享环境。
输入媒体和模型只读，各任务独立输出/缓存；4090按实际负载共享，在Issue维护任务占用。
