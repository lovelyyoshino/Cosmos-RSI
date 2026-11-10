# 上游来源与许可

复现版本以 [third_party/PINS.md](../third_party/PINS.md) 为准。安装时先核对固定 SHA，再阅读对应版本的文档；下表提供源仓库入口。

| 组件 | 官方来源 | 需要阅读的内容 |
| --- | --- | --- |
| starVLA | https://github.com/starVLA/starVLA | 稳定分支 `starVLA`；`docs/agent_skills/integrate-starvla-dataset/`；CosmosGR00T 与 CosmosReason2 接口 |
| Cosmos-Reason2 | https://github.com/nvidia-cosmos/cosmos-reason2 | 模型加载、vLLM 与 CUDA Graph；权重 https://huggingface.co/nvidia/Cosmos-Reason2-2B |
| Cosmos-Transfer2.5 | https://github.com/nvidia-cosmos/cosmos-transfer2.5 | `docs/setup.md`、`docs/inference.md`；Jetson AGX 的 cu130 安装与 distilled edge 参数 |
| Cosmos-Predict2.5 | https://github.com/nvidia-cosmos/cosmos-predict2.5 | LIBERO 动作条件分支；配合本仓库 `patches/cosmos-predict2.5/` 阅读 |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | 原始示范、任务定义、资产和仿真环境 |
| LIBERO-plus | https://github.com/sylvestf/LIBERO-plus | 扰动场景和资产；与 LIBERO 使用独立导入配置 |
| Isaac-GR00T | https://github.com/NVIDIA/Isaac-GR00T | 备用策略和独立延迟参照，不是闭环默认策略 |

本仓库不捆绑上游完整源码、模型权重或数据集。使用上游代码与补丁前检查相应版本的 LICENSE；下载模型和数据前确认账号许可、使用条款及再分发限制。镜像下载记录真实来源、revision 和逐文件摘要，不代表已取得原站门控许可。

补丁的固定基线与作用见 [PINS.md](../third_party/PINS.md) 和 [Transfer 适配说明](../patches/cosmos-transfer2.5/README.md)。不要将本项目 MIT 许可解释为覆盖上游代码、模型或数据。
