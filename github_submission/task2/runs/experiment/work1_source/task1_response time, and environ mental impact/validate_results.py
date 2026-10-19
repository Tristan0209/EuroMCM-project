import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import ROOT


def verify_outputs():
    output, processed = ROOT / "outputs", ROOT / "data/processed"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    config, profile = manifest["config"], manifest["active_profile"]
    runs = pd.read_csv(output / "simulation_runs.csv")
    summary = pd.read_csv(output / "simulation_summary.csv")
    annual = pd.read_csv(processed / "annual_carbon_factors.csv").set_index("scenario")
    group_keys = ["workload", "rate_multiplier", "replicas", "carbon_scenario"]
    workloads = sorted(manifest["input_transformations"])
    expected = len(workloads) * len(config["arrival_rate_multipliers"]) * len(config["replicas"]) * len(config["carbon_scenarios"]) * config["repetitions"]
    assert len(runs) == expected
    assert not runs[group_keys + ["repeat"]].duplicated().any()
    assert len(summary) * config["repetitions"] == expected
    assert np.isfinite(runs[["mean_latency_s", "p95_latency_s", "facility_kwh", "emissions_g"]]).all().all()
    assert (runs[["mean_latency_s", "p95_latency_s", "facility_kwh", "emissions_g"]] > 0).all().all()
    assert ((runs.late_fraction >= 0) & (runs.late_fraction <= 1)).all()
    assert (runs.p95_ttft_s <= runs.p95_latency_s + 1e-8).all()
    assert ((runs.utilization_observed >= 0) & (runs.utilization_observed <= 1 + 1e-8)).all()
    assert runs.service_provenance.eq(profile["service_provenance"]).all()
    assert runs.power_provenance.eq(profile["power_provenance"]).all()
    np.testing.assert_allclose(runs.facility_kwh, runs.it_kwh * profile["pue"], rtol=1e-9)
    np.testing.assert_allclose(runs.emissions_g_per_1000_requests, 1000 * runs.emissions_g / runs.requests, rtol=1e-9)
    np.testing.assert_allclose(runs.facility_kwh_per_1000_requests, 1000 * runs.facility_kwh / runs.requests, rtol=1e-9)
    static_errors = []
    for scenario, group in runs.groupby("carbon_scenario"):
        if scenario not in annual.index:
            continue
        factor = annual.loc[scenario]
        assert group.carbon_unit.eq(factor.unit).all()
        relative = np.abs(group.emissions_g / (group.facility_kwh * factor.intensity) - 1)
        static_errors.extend(relative.tolist())
        assert relative.max() < 1e-9
    values = ["mean_latency_s", "p95_latency_s", "p95_ttft_s", "late_fraction", "facility_kwh", "emissions_g",
              "emissions_g_per_1000_requests", "utilization_observed", "offered_load", "drain_s"]
    for _, row in summary.iterrows():
        mask = np.ones(len(runs), dtype=bool)
        for key in group_keys:
            mask &= runs[key].to_numpy() == row[key]
        part = runs[mask]
        assert len(part) == config["repetitions"]
        for field in values:
            assert np.isclose(part[field].mean(), row[field + "_mean"], rtol=1e-8, atol=1e-10)
            assert row[field + "_ci95_low"] <= row[field + "_mean"] + 1e-10
            assert row[field + "_mean"] <= row[field + "_ci95_high"] + 1e-10
    schedule = pd.read_csv(output / "example_request_schedule.csv")
    timeline = pd.read_csv(output / "example_power_timeline.csv")

    example_workload = "azure_conversation"
    example = runs[(runs.workload == example_workload) & (runs.rate_multiplier == 1) & (runs.replicas == 1)
                   & (runs["repeat"] == 0) & (runs.carbon_scenario == "EEA_EU27_2024")].iloc[0]
    assert len(schedule) == example.requests and not schedule.request_id.duplicated().any()
    assert (schedule.queue_wait_s >= -1e-9).all()
    np.testing.assert_allclose(schedule.start_s - schedule.arrival_s, schedule.queue_wait_s, atol=1e-8)
    np.testing.assert_allclose(schedule.finish_s - schedule.start_s, schedule.service_s, atol=1e-8)
    np.testing.assert_allclose(schedule.latency_s, config["network_control_s"] + schedule.finish_s - schedule.arrival_s, atol=1e-8)
    assert (schedule.ttft_s <= schedule.latency_s + 1e-8).all()
    for _, replica in schedule.groupby("replica"):
        replica = replica.sort_values("start_s")
        assert np.all(replica.start_s.to_numpy()[1:] >= replica.finish_s.to_numpy()[:-1] - 1e-8)
    assert np.isclose(schedule.latency_s.quantile(0.95), example.p95_latency_s, rtol=1e-9)
    duration = timeline.end_s - timeline.start_s
    assert (duration > 0).all()
    assert np.all((timeline.busy_replicas >= 0) & (timeline.busy_replicas <= 1))
    np.testing.assert_allclose(timeline.it_power_w, profile["shared_w"] + profile["gpu_count_per_server"] * profile["gpu_idle_w"] + (profile["gpu_busy_w"] - profile["gpu_idle_w"]) * timeline.busy_replicas)
    np.testing.assert_allclose(timeline.it_kwh, timeline.it_power_w * duration / 3.6e6, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(timeline.facility_kwh, timeline.it_kwh * profile["pue"], rtol=1e-8, atol=1e-14)
    assert np.isclose(timeline.facility_kwh.sum(), example.facility_kwh, rtol=1e-9)
    conserved = (profile["shared_w"] + profile["gpu_count_per_server"] * profile["gpu_idle_w"]) * timeline.end_s.max() + (profile["gpu_busy_w"] - profile["gpu_idle_w"]) * schedule.service_s.sum()
    conservation_error = abs(timeline.it_kwh.sum() * 3.6e6 / conserved - 1)
    assert conservation_error < 1e-9


    gb = pd.read_csv(processed / "gb_carbon_intervals.csv")
    boundaries = np.r_[gb.start_s.iloc[0], gb.end_s.to_numpy()]
    cumulative = np.r_[0, np.cumsum((gb.end_s - gb.start_s) * gb.intensity)]
    assert timeline.end_s.max() <= boundaries[-1]
    integrated_intensity = np.interp(timeline.end_s, boundaries, cumulative) - np.interp(timeline.start_s, boundaries, cumulative)
    independent_gb = float(np.sum(timeline.it_power_w * profile["pue"] * integrated_intensity) / 3.6e6)
    observed_gb = runs[(runs.workload == example_workload) & (runs.rate_multiplier == 1) & (runs.replicas == 1)
                       & (runs["repeat"] == 0) & (runs.carbon_scenario == "GB_NESO_2024-05-12")].iloc[0]
    dynamic_error = abs(independent_gb / observed_gb.emissions_g - 1)
    assert dynamic_error < 1e-9
    return {"simulation_rows_checked": len(runs), "summary_rows_checked": len(summary),
            "example_request_rows_checked": len(schedule), "example_power_intervals_checked": len(timeline),
            "max_static_carbon_relative_error": max(static_errors),
            "example_energy_conservation_relative_error": conservation_error,
            "independent_dynamic_carbon_relative_error": dynamic_error,
            "source_of_server_parameters": profile["service_provenance"],
            "power_parameter_source": profile["power_provenance"]}


def main():
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "saved-result consistency and independent arithmetic checks; not measured real-system prediction accuracy"}
    try:
        report.update({"passed": True, "details": verify_outputs()})
    except Exception as error:
        report.update({"passed": False, "error_type": type(error).__name__, "error": str(error)})
    (ROOT / "outputs/result_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
