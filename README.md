# EuroMCM-project

当前第二问只保留最终的 **Pre-LightGBM质量预测＋成本阈值分流＋任务风险回退** 版本，入口为[task2_complete](task2_complete/README_zh.md)。

- [论文HTML阅读版](task2_complete/paper_pre_lightgbm/问题二_Pre-LightGBM论文阅读版.html)：公式和图片内嵌。
- [论文MD正文](task2_complete/paper_pre_lightgbm/问题二_Pre-LightGBM论文正文.md)与[逐步讲解](task2_complete/paper_pre_lightgbm/问题二_逐步讲解.md)。
- [最终运行报告](task2_complete/runs/work1_integrated_20261016/RESULTS_zh.md)。

运行最终路由，不执行付费LLM调用：

```powershell
python -B task2_complete/predict.py --benchmark hellaswag --prompt "Choose the best continuation."
```

Work 1、AI_datasets及第一问的latency_environment材料保留。旧第二问独立方案目录已删除；当前对照实验必需的类、编码器和RF缓存集中到task2_complete/support。清理记录见task2_complete/cleanup_final_20261017.json。
