import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from core import CarbonSeries, integrate_energy, metrics, predict_service, schedule
from common import ROOT


def load_carbon(name, horizon, annual, gb):
    if name == "GB_NESO_2024-05-12":
        return CarbonSeries(gb.start_s.to_numpy(), gb.end_s.to_numpy(), gb.intensity.to_numpy(),
                            "NESO_actual_GB_2024-05-12_to_19", "gCO2/kWh",
                            "GB_operational_CO2_not_CO2e_or_marginal")
    rows = annual[annual.scenario == name]
    if len(rows) != 1:
        raise ValueError("Unknown or duplicate annual scenario: %s" % name)
    row = rows.iloc[0]
    return CarbonSeries([0.0], [max(horizon + 1, 1.0)], [float(row.intensity)],
                        name, row.unit, row.boundary)


def aggregate(frame):
    keys = ["workload", "rate_multiplier", "replicas", "carbon_scenario", "carbon_unit", "carbon_boundary"]
    values = ["mean_latency_s", "p95_latency_s", "p95_ttft_s", "late_fraction",
              "facility_kwh", "emissions_g", "emissions_g_per_1000_requests",
              "utilization_observed", "offered_load", "drain_s"]
    records = []
    for group_key, group in frame.groupby(keys, sort=True):
        record = dict(zip(keys, group_key))
        record["repetitions"] = len(group)
        record["interval_interpretation"] = "conditional_MC_mean_across_service_noise_seeds_not_empirical_accuracy"
        for name in values:
            sample = group[name].to_numpy(dtype=float)
            mean = float(sample.mean())
            half_width = float(student_t.ppf(0.975, len(sample) - 1) * sample.std(ddof=1) / np.sqrt(len(sample))) if len(sample) > 1 else 0.0
            record[name + "_mean"] = mean
            record[name + "_ci95_low"] = mean - half_width
            record[name + "_ci95_high"] = mean + half_width
        records.append(record)
    result = pd.DataFrame(records)


    result["pareto_efficient"] = True
    for _, indices in result.groupby(["workload", "rate_multiplier", "carbon_scenario"]).groups.items():
        for index in indices:
            a = result.loc[index]
            for other in indices:
                b = result.loc[other]
                no_worse = b.p95_latency_s_mean <= a.p95_latency_s_mean and b.emissions_g_per_1000_requests_mean <= a.emissions_g_per_1000_requests_mean
                strictly_better = b.p95_latency_s_mean < a.p95_latency_s_mean or b.emissions_g_per_1000_requests_mean < a.emissions_g_per_1000_requests_mean
                if no_worse and strictly_better:
                    result.loc[index, "pareto_efficient"] = False
                    break
    return result


def figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted(summary.workload.unique())
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.5), squeeze=False, dpi=160)
    selected_name = "EEA_EU27_2024" if (summary.carbon_scenario == "EEA_EU27_2024").any() else summary.carbon_scenario.iloc[0]
    selected = summary[summary.carbon_scenario == selected_name]
    if selected.empty:
        selected = summary[summary.carbon_scenario == summary.carbon_scenario.iloc[0]]
    for ax, workload in zip(axes[0], names):
        for replica, group in selected[selected.workload == workload].groupby("replicas"):
            group = group.sort_values("rate_multiplier")
            ax.plot(group.rate_multiplier, group.p95_latency_s_mean, "o-", label="%d replica(s)" % replica)
        ax.set(xlabel="Arrival-rate multiplier (common workload)", ylabel="P95 end-to-end latency (s)", title=workload)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Queue simulation with assumed or fitted profile; see run_manifest.json", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "latency_vs_load.png")
    plt.close(fig)

    subset = selected[selected.rate_multiplier == 1.0]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    for workload, group in subset.groupby("workload"):
        ax.plot(group.p95_latency_s_mean, group.emissions_g_per_1000_requests_mean, "o-", label=workload)
        for _, row in group.iterrows():
            ax.annotate("c=%d" % row.replicas, (row.p95_latency_s_mean, row.emissions_g_per_1000_requests_mean), xytext=(4, 5), textcoords="offset points", fontsize=9)
    gas = "CO2e" if selected.carbon_unit.iloc[0] == "gCO2e/kWh" else "CO2"
    ax.set(xlabel="P95 end-to-end latency (s)", ylabel="Operational g%s / 1,000 requests" % gas,
           title="Profile-dependent simulation, %s" % selected_name)
    ax.margins(x=0.12, y=0.12)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "latency_carbon_tradeoff.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile = config["profile"]
    if profile["service_boundary"] == "isolated_client_includes_transport" and config["network_control_s"] != 0:
        raise ValueError("Client-measured service already includes transport: set network_control_s=0")
    data, output = ROOT / "data/processed", ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(data / "annual_carbon_factors.csv")
    gb = pd.read_csv(data / "gb_carbon_intervals.csv")
    rows, input_audit = [], {}
    sample_saved = False
    for workload in ["azure_conversation", "azure_code"]:
        requests = pd.read_csv(data / (workload + ".csv")).iloc[:config["max_requests_per_trace"]].copy()
        deterministic_first, deterministic_service = predict_service(requests, profile, stochastic=False)
        arrivals = requests.arrival_s.to_numpy(dtype=float)
        arrivals = arrivals - arrivals[0]
        span = float(arrivals[-1])
        if len(arrivals) < 2 or span <= 0:
            raise ValueError("A trace needs at least two distinct arrival times")


        reference_span = (len(arrivals) - 1) * float(deterministic_service.mean()) / config["reference_utilization"]
        time_scale = reference_span / span
        input_audit[workload] = {"requests": len(arrivals), "original_span_s": span,
                                 "time_scale_at_1x": time_scale,
                                 "note": "time-scaled hypothetical demand; original joint token/arrival structure retained"}
        domain = profile.get("calibration_domain")
        if domain:
            outside = (requests.input_tokens < domain["min_input_tokens"]) | (requests.input_tokens > domain["max_input_tokens"])
            outside |= (requests.output_tokens < domain["min_output_tokens"]) | (requests.output_tokens > domain["max_output_tokens"])
            input_audit[workload]["requests_outside_calibration_domain"] = int(outside.sum())
        for repeat in range(config["repetitions"]):
            seed = config["seed"] + repeat
            first, service = predict_service(requests, profile, seed=seed)
            for rate in config["arrival_rate_multipliers"]:
                if rate <= 0:
                    raise ValueError("Rate multipliers must be positive")
                actual_arrivals = arrivals * time_scale / rate
                interval = actual_arrivals[-1] / (len(arrivals) - 1)
                window_end = actual_arrivals[-1] + interval
                for replica in config["replicas"]:
                    trace = schedule(actual_arrivals, service, first, replica, config["network_control_s"])
                    offered_load = float(service.mean()) / interval / replica
                    horizon = max(window_end, float(trace.finish_s.max()))
                    for carbon_name in config["carbon_scenarios"]:
                        carbon = load_carbon(carbon_name, horizon, annual, gb)
                        energy, timeline = integrate_energy(trace, replica, profile["gpu_idle_w"], profile["gpu_busy_w"], profile["pue"], carbon, window_end, shared_w=profile["shared_w"], gpu_count=profile["gpu_count_per_server"], server_max_w=profile["server_max_w"])
                        record = metrics(trace, energy, config["deadline_s"])
                        record.update({"workload": workload, "replicas": replica,
                                       "rate_multiplier": rate, "repeat": repeat, "seed": seed,
                                       "carbon_scenario": carbon_name, "offered_load": offered_load,
                                       "stationary_stability_necessary_condition": offered_load < 1,
                                       "time_scale": time_scale / rate,
                                       "service_provenance": profile["service_provenance"],
                                       "power_provenance": profile["power_provenance"],
                                       "power_scope": profile["power_scope"], "shared_w": profile["shared_w"], "gpu_idle_w": profile["gpu_idle_w"], "gpu_busy_w": profile["gpu_busy_w"]})
                        rows.append(record)
                        if not sample_saved and rate == 1 and replica == 1 and carbon_name == "EEA_EU27_2024":
                            original = requests.reset_index(drop=True).rename(columns={"arrival_s": "original_arrival_s", "timestamp": "original_timestamp"})
                            pd.concat([original, trace], axis=1).to_csv(output / "example_request_schedule.csv", index=False)
                            timeline.to_csv(output / "example_power_timeline.csv", index=False)
                            sample_saved = True
    results = pd.DataFrame(rows)
    results.to_csv(output / "simulation_runs.csv", index=False)
    summary = aggregate(results)
    summary.to_csv(output / "simulation_summary.csv", index=False)
    manifest = {
        "config": config, "active_profile": profile, "input_transformations": input_audit,
        "scope": "finite-horizon FCFS independent replicas; includes all arrival-window requests and drain energy",
        "warmup": "empty initial queue; intentional finite-workload experiment, not a steady-state estimate",
        "empirical_queue_validation": "NOT performed without matching measured service-system logs",
        "intervals": "95% t intervals across conditional service-noise replications; no claim about structural model error",
        "carbon_alignment": "GB factors from 2024-05-12; time-scaled workload replay is a scenario, not historical reconstruction",
        "batching": "not modeled; max_num_seqs and GPU count are never treated as replica count",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    figures(summary, output)
    print("Completed %d runs, %d summary rows" % (len(results), len(summary)))
    print("Service:", profile["service_provenance"], "; power:", profile["power_provenance"])


if __name__ == "__main__":
    main()
