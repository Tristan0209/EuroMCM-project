# Inputs and provenance

保留的文件来自原项目及其既有工作副本，没有新增或改写实验观测值。

| 输入 | 保存位置 | 用途 |
| --- | --- | --- |
| RouterBench | `AI_datasets/RouterBench/routerbench_0shot.csv` | 第一问质量模型与第二问训练和来源核查 |
| 成对请求、结果及固定特征 | `task2/inputs/` | 固定训练划分、训练特征和离线结果评估 |
| Azure 处理后片段 | `Work 1/task1_response time, and environ mental impact/data/processed/` | 请求到达及长度模型 |
| 电力碳强度 | `TASK 3/data_preparation/Carbon Intensity/` | 区域碳强度输入 |
| EEA、Ember、NESO、ML.ENERGY 等 | `AI_datasets/` 及处理后输入中的来源字段 | 环境参数及公开测量对照 |

Azure 原始完整文件体积较大，未复制到提交包。完整重建原始预处理时，将原项目 `AI_datasets/Azure/` 中以下文件放入仓库的同名目录：

- `AzureLLMInferenceTrace_code_1week.csv`
- `AzureLLMInferenceTrace_conv_1week.csv`

处理后数据的来源和哈希见其 `data_audit.json`。来源文件的许可及再分发条件以发布方为准；此提交包没有为外部数据重新授予许可。

最终第四问所需 Spatial 求解器来自原项目既有工作副本的 `TASK 3/task3_model/results/robustness_v2/final_release/`；其公共基础源码、配置和 `results/data` 与本地第三问副本逐文件核对一致。随包添加了该求解器及必要的邻接模块和固定配置，原冻结清单仍然保留。

第三问迁移检查仍依赖旧版的两张对照表，因此只保留 `archives/legacy_power_175_250_20261017/results/data/` 下的 `requests.csv` 和 `evaluation_only_historical_quality.csv`，其余旧归档不收录。
