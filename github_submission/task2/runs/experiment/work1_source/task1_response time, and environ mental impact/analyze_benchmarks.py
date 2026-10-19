import json

import numpy as np
import pandas as pd

from common import ROOT, DATASETS, sha256


def main():
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    factors = pd.read_csv(ROOT / "data/processed/annual_carbon_factors.csv")
    eea = factors[factors.scenario == "EEA_EU27_2024"]
    rows = []
    existing = DATASETS / "ML.ENERGY"
    local_files = sorted(p for p in existing.glob("*") if p.suffix.lower() == ".json" and p.stem in {"Qwen3-8B", "Qwen3-14B", "Qwen3-32B"})
    source_files = local_files
    if len(local_files) != 3:
        raise ValueError("Expected all three downloaded Qwen3 measurement JSON files")
    for path in source_files:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for index, config in enumerate(data["configurations"]):
            energy_j = config.get("energy_per_request_joules")
            record = {
                "model_id": data["model_id"], "task": data["task"],
                "precision": data["weight_precision"], "configuration_index": index,
                "gpu_model": config["gpu_model"], "num_gpus": config["num_gpus"],
                "max_num_seqs": config["max_num_seqs"], "avg_batch_size": config.get("avg_batch_size"),
                "avg_output_tokens": config.get("avg_output_len"),
                "median_itl_ms": config["median_itl_ms"], "p95_itl_ms": config["p95_itl_ms"],
                "output_throughput_tokens_per_s": config["output_throughput_tokens_per_sec"],
                "avg_power_watts_reported": config["avg_power_watts"],
                "gpu_joules_per_output_token": config["energy_per_token_joules"],
                "gpu_joules_per_request": energy_j,
                "gpu_kwh_per_1000_requests": float(energy_j) / 3600 if energy_j is not None else np.nan,
                "measurement_scope": "published_GPU_energy_not_complete_server_or_facility",
                "latency_scope": "ITL_not_end_to_end_or_TTFT",
                "source_file": str(path.resolve()),
            }
            if len(eea) == 1 and energy_j is not None:
                record["gpu_electricity_gco2e_per_1000_EEA_EU2024"] = energy_j / 3600 * float(eea.intensity.iloc[0])
            if config["num_gpus"] == 1:
                calculated = config["energy_per_token_joules"] * config["output_throughput_tokens_per_sec"]
                record["single_gpu_power_identity_relative_error"] = abs(calculated / config["avg_power_watts"] - 1)
            rows.append(record)
    if not rows:
        raise FileNotFoundError("No measurement JSONs found in AI_datasets/ML.ENERGY")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "public_measured_benchmarks.csv", index=False)
    report = {
        "configuration_rows": len(frame), "models": sorted(frame.model_id.unique()),
        "source_files": [str(p.resolve()) for p in source_files],
        "source_sha256": {str(p.resolve()): sha256(p) for p in source_files},
        "observed_metrics": "ITL, throughput, GPU energy, reported average power",
        "not_available_in_these_summaries": ["per-request end-to-end duration", "queue delay", "TTFT", "idle server power", "CPU/RAM energy"],
        "not_fitted": "No full-request service model or server power is fitted from these summaries",
        "consistency_check": "E_per_token * throughput approximately equals GPU power for single-GPU records; not independent empirical validation",
        "derived_carbon": "Published GPU kWh times EEA annual factor; no PUE or missing host energy is silently added",
    }
    (output / "public_benchmark_scope.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hardware = sorted(frame.gpu_model.unique())
    fig, axes = plt.subplots(1, len(hardware), figsize=(6 * len(hardware), 4.7), squeeze=False, dpi=160)
    for ax, gpu in zip(axes[0], hardware):
        for (model, count), group in frame[frame.gpu_model == gpu].groupby(["model_id", "num_gpus"]):
            ax.scatter(group.median_itl_ms, group.gpu_joules_per_output_token, s=35, alpha=0.8,
                       marker="o" if count == 1 else "^", label="%s (%d GPU)" % (model.split("/")[-1], count))
        ax.set(xlabel="Measured median inter-token latency (ms)", ylabel="Measured GPU joules / output token",
               title="Published benchmark: %s" % gpu)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Measured configurations; ITL is not end-to-end request latency", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "public_measured_itl_energy.png")
    plt.close(fig)
    print("Analyzed %d measured configurations for %d models" % (len(frame), frame.model_id.nunique()))


if __name__ == "__main__":
    main()
