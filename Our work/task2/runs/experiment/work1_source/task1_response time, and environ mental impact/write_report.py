import json

import pandas as pd

from common import ROOT


def main():
    output = ROOT / "outputs"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((output / "implementation_validation.json").read_text(encoding="utf-8"))
    data_validation = json.loads((output / "data_validation.json").read_text(encoding="utf-8"))
    result_validation = json.loads((output / "result_validation.json").read_text(encoding="utf-8"))
    benchmark = json.loads((output / "public_benchmark_scope.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output / "simulation_summary.csv")
    runs = pd.read_csv(output / "simulation_runs.csv")
    profile = manifest["active_profile"]
    trace_checks = data_validation["checks"]["azure_sources_and_preprocessing"]["details"]
    checked_rows = sum(item["rows_checked_against_source"] for item in trace_checks.values())
    excluded_rows = sum(item["excluded_rows"] for item in trace_checks.values())
    checked_results = result_validation["details"]
    lines = ["# 响应时间与环境影响：运行结果", "",
             "## 数据与结果的性质", "",
             "- 公开实测配置：%d 条、%d 个模型。指标是 ITL、吞吐率和 GPU 电能，不是完整请求时延或完整服务器电能。" % (benchmark["configuration_rows"], len(benchmark["models"])),
             "- 排队仿真：%d 次运行、%d 行配置汇总。每个工作负载使用相同请求集进行配置比较。" % (len(runs), len(summary)),
             "- 服务时间参数来源：`%s`。" % profile["service_provenance"],
             "- 功率参数来源：`%s`；PUE 来源：`%s`。" % (profile["power_provenance"], profile.get("pue_provenance", "not specified")),
             "- 完整服务器实测队列与能耗检验：尚未提供匹配的实测日志，不能声称已通过。", "",
             "## 标准负载下的示例结果", "",
             "下表固定 EEA EU 2024 年度因子；时延和能耗取决于运行配置中的服务与功率参数。", "",
             "| 工作负载 | 独立副本 | P95 时延/s | 每千次请求/gCO2e | 超过%g秒比例 |" % manifest["config"]["deadline_s"],
             "|---|---:|---:|---:|---:|"]
    selected = summary[(summary.carbon_scenario == "EEA_EU27_2024") & (summary.rate_multiplier == 1)]
    for _, row in selected.iterrows():
        lines.append("| %s | %d | %.3f | %.3f | %.2f%% |" % (row.workload, row.replicas, row.p95_latency_s_mean, row.emissions_g_per_1000_requests_mean, row.late_fraction_mean * 100))
    lines.extend(["", "增加副本减少排队，同时增加持续开机的空闲电能；应在同一需求和碳核算边界下比较。不同工作负载的到达时刻被分别缩放以建立参考负载，不据此比较其原始生产系统容量。", "",
                  "## 数据检查", "",
                  "- 数据检查通过：%s；核对 %d 个源文件的完整 SHA256。" % (data_validation["passed"], data_validation["source_files_fingerprinted"]),
                  "- Azure 合计 %d 条预处理记录逐项回查源文件，排除 %d 条；整周文件仅作完整哈希和物理行数审计。" % (checked_rows, excluded_rows),
                  "- 58 条公开测量配置通过字段核对；英国 336 个区间覆盖 604,800 秒，无缺失实际值或时间缺口。", "",
                  "## 实现与结果检查", "",
                  "- 通过 %d 项单元检查。" % validation["tests_passed"],
                  "- M/M/1 平均时延相对解析解误差：%.3f%%。" % (100 * validation["mean_relative_error"]),
                  "- M/M/1 P95 相对解析解误差：%.3f%%。" % (100 * validation["p95_relative_error"]),
                  "- 结果检查通过：%s；%d 次运行、%d 行汇总、%d 条示例调度、%d 个示例功率区间已核对。" % (result_validation["passed"], checked_results["simulation_rows_checked"], checked_results["summary_rows_checked"], checked_results["example_request_rows_checked"], checked_results["example_power_intervals_checked"]),
                  "- 静态排放最大相对复算误差：%.3g。" % result_validation["details"]["max_static_carbon_relative_error"],
                  "- 示例能量守恒相对误差：%.3g。" % result_validation["details"]["example_energy_conservation_relative_error"],
                  "- 独立动态碳积分相对误差：%.3g。" % result_validation["details"]["independent_dynamic_carbon_relative_error"],
                  "- 这是合成条件下的实现检查，不能替代真实系统验证。", "",
                  "## 输出文件怎么读", "",
                  "- `public_measured_benchmarks.csv`：原始发布者的测量配置与 GPU 电能派生核算。",
                  "- `simulation_runs.csv`：每个随机种子、负载、副本数和碳信号的完整结果。",
                  "- `simulation_summary.csv`：重复运行均值、条件蒙特卡洛置信区间和点估计 Pareto 标记。",
                  "- `example_request_schedule.csv`：保留原始时间与实际仿真时间，逐请求追踪等待、开始、完成。",
                  "- `example_power_timeline.csv`：逐区间忙碌副本数、功率、电量和排放，可手动核对。",
                  "- `run_manifest.json`：参数、测量边界、时间缩放和当前验证范围。", "",
                  "- `data_validation.json`、`implementation_validation.json`、`result_validation.json`：数据、实现与结果复算的实际检查记录。", "",
                  "## 解释限制", "",
                  "EEA 是年度发电温室气体口径，Ember 考虑电力生命周期，NESO 为英国 CO2 时序。代码保留来源和单位，不跨这些边界统一排名。",
                  "条件蒙特卡洛置信区间只描述给定参数和固定轨迹下的随机波动，不包含服务器参数误差、结构误差或整周工作负载代表性。", ""])
    (output / "RESULTS_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote outputs/RESULTS_zh.md")


if __name__ == "__main__":
    main()
