# 上游版本锁定

锁定时间：2026-11-06T18:04:44.259821+08:00；机器：Thor。已应用补丁见表，LFS 大文件按需下载。

| 仓库 | 分支 | SHA | 克隆日期 | 本地 patch |
| --- | --- | --- | --- | --- |
| https://github.com/NVIDIA/Isaac-GR00T.git | main | `7b304f95b95790490b79bd263ba57015c3c12dfd` | 2026-11-06T18:04:44.259821+08:00 | 无 |
| https://github.com/Lifelong-Robot-Learning/LIBERO.git | master | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | 2026-11-06T18:04:44.259821+08:00 | 无 |
| https://github.com/sylvestf/LIBERO-plus.git | main | `4976dc30028e805ff8094b55501d532c48fec182` | 2026-11-06T18:04:44.259821+08:00 | 无 |
| https://github.com/nvidia-cosmos/cosmos-transfer2.5.git | main | `2ff49d0717af02057ae79bc75c00fbff9da1b4e7` | 2026-11-06T18:04:44.259821+08:00 | `patches/cosmos-transfer2.5/0001-local-resource-paths.patch`、`patches/cosmos-transfer2.5/0002-guardrail-fail-closed.patch`、`patches/cosmos-transfer2.5/0003-experiment-overrides.patch`、`patches/cosmos-transfer2.5/0005-local-wheel-transport.patch`、`patches/cosmos-transfer2.5/0006-transfer-only-registry.patch`、`patches/cosmos-transfer2.5/0007-siglip2-local-path.patch`、`patches/cosmos-transfer2.5/0008-i2v-attention-backend.patch` |
| https://github.com/starVLA/starVLA.git | starVLA | `3422b9f2387b6f682cf02802904a77b23ab13afd` | 2026-11-06T18:04:44.259821+08:00 | `patches/starVLA/0001-lazy-pytorch3d-rotation.patch` |

2026-11-07接续：Transfer同一SHA另外应用 `patches/cosmos-transfer2.5/0009-local-multicontrol-checkpoints.patch`，只为原生四分支传入已核验的本地权重，不改变采样、控制权重、安全检查或像素。

2026-11-08数据交接：starVLA同一稳定分支SHA应用 `patches/starVLA/0001-lazy-pytorch3d-rotation.patch`。仅将旋转转换库的导入推迟到真正的RotationTransform构造时；原始动作/图像读取无需该库，实际旋转转换仍要求真实依赖。未更改模型前向、动作语义、采样、归一化、图像和历史权重。

2026-11-09新增Predict动作后果支路：Thor的 `~/rsi/third_party/cosmos-predict2.5` 固定为官方仓库 `https://github.com/nvidia-cosmos/cosmos-predict2.5.git`、SHA `a2c298b0a3df3778b973fe65e9e58877b292d8a7`（detached HEAD）。代码已取得；权重、接口适配与运行证据单独记录，不据克隆成功宣称模型已运行。

Predict同一SHA新增 `patches/cosmos-predict2.5/0001-explicit-libero-action-condition.patch`：显式将已归一化的连续16×7候选动作写入LIBERO动作条件槽4，要求五个干净条件槽。不沿用上游被忽略的 `previous_generated_latent` 参数。`0002-defer-optional-flash-attention-check.patch` 将可选FlashAttention依赖断言移至实际FlashAttention类构造；离线预计算语言条件与Torch SDPA预测无需实例化该类，在线选择FlashAttention时仍明确检查依赖。运行结果及适用边界另行记录。
