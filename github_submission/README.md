# EuroMCM models and experiments

本仓库包含四问的模型实现、必要输入、已训练模型和保存的实验结果。目录名称保留与程序相同的空格及大小写。

| 目录 | 内容与入口 |
| --- | --- |
| `Work 1/` | 质量预测、回答评估、成本及延迟和碳排放模型 |
| `task2/` | 自适应路由；`predict.py` 用于已训练模型推断，`run_all.py` 用于重新训练 |
| `TASK 3/` | 数据准备、离散事件仿真与调度优化；`task3_model/run_all.py` 是分阶段入口 |
| `TASK 4/` | 四情景评估；`simple_assessment/run_simple.py` 是最终实验入口 |
| `AI_datasets/` | 必要的原始输入及外部资料；详情见 `DATA_SOURCES.md` |

从仓库根目录执行以下命令。本次验证使用 Python 3.9.13；训练模型能否跨版本加载还取决于 joblib、scikit-learn、LightGBM 和 PyTorch 的版本。原实验环境信息保留在结果中的 audit 文件内。

```powershell
python -m pip install -r requirements.txt
python -B validate_submission.py
python -B task2/test_complete.py
python -B task2/predict.py --benchmark hellaswag --prompt "Choose the best continuation."
```

第三问单元测试：

```powershell
Push-Location "TASK 3/task3_model"
python -B -m pytest -q -p no:cacheprovider tests
Pop-Location
```

第四问边界检查（使用尚不存在的新输出目录）：

```powershell
python -B "TASK 4/tests/run_contracts.py" --out "TASK 4/audit/submission_contracts"
```

第四问完整复现使用固定的四个情景及三组种子；运行会产生新的结果，不增加已有实验的样本量。执行耗时取决于计算环境。新结果写入 `reproduced_dry_run`、`reproduced_runs`，历史 `runs/` 保持作为原始实验记录。

```powershell
python -B "TASK 4/simple_assessment/run_simple.py" dry-run
python -B "TASK 4/simple_assessment/run_simple.py" run
python -B "TASK 4/simple_assessment/run_simple.py" validate
```

第二问重新训练应指定全新的输出目录：

```powershell
python -B task2/run_all.py --source-project . --out task2/runs/reproduction
```

第一问从原始 Azure 轨迹重新预处理，以及第二问完整验证原始轨迹，都需要额外补齐 `DATA_SOURCES.md` 所列的两份大体积 Azure CSV。已有处理后输入及保存结果可以直接检查。

当前第二问结果位于 `task2/runs/experiment/`；第三问数据与结果位于 `TASK 3/task3_model/results/`；第四问汇总表位于 `TASK 4/simple_assessment/Task4_Simple_Results.csv`，逐次执行记录位于同目录下 `runs/`。

第三问最终 Spatial 求解器是第四问的依赖，随包保存在 `TASK 3/task3_model/results/robustness_v2/final_release/`。该目录中的最终对照表可以检查；其上游工作副本中的逐次运行轨迹未重复收录。当前 `TASK 3/task3_model/results/runs/` 对应基础调度模型的历史运行，不应与最终 Spatial 版本混淆。

代码注释和 Python 文档字符串已移除；执行逻辑通过抽象语法树比较校验。提交版只额外调整了路径、默认运行目录、目录输入支持及校验入口。原始实验清单中的哈希、日期、来源说明和限制保留原值；它们标识原实验文件，不代表整理后的源文件。提交版校验使用 `SUBMISSION_MANIFEST.json` 和具名的 `submission_*manifest.json`。历史记录中的绝对路径仅作为原环境信息，不能据此认为新的运行已经完成。

本仓库的代码整理不构成模型正确性、论文一致性或比赛规则合规性的完整认证。公开数据及模型的来源与适用许可需要遵循各自发布方要求。
