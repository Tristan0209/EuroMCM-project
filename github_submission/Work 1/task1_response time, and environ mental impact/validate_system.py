import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT


def compare_requests(predicted, measured):
    for name, frame in [("predicted", predicted), ("measured", measured)]:
        if not {"request_id", "latency_s"}.issubset(frame.columns):
            raise ValueError("%s CSV lacks request_id or latency_s" % name)
        if frame.request_id.isna().any() or frame.request_id.duplicated().any():
            raise ValueError("Request IDs must be present and unique")
        if not np.isfinite(frame.latency_s).all() or (frame.latency_s <= 0).any():
            raise ValueError("Latency must be positive and finite")
    if set(predicted.request_id) != set(measured.request_id):
        raise ValueError("Predicted and observed request sets differ; no silent row dropping")
    joined = predicted.merge(measured, on="request_id", validate="one_to_one", suffixes=("_predicted", "_measured"))
    error = joined.latency_s_predicted - joined.latency_s_measured
    reference = float(joined.latency_s_measured.quantile(0.95))
    result = {"matched_requests": len(joined), "latency_mae_s": float(np.abs(error).mean()),
              "latency_rmse_s": float(np.sqrt(np.mean(error ** 2))), "latency_bias_s": float(error.mean()),
              "predicted_p95_s": float(joined.latency_s_predicted.quantile(0.95)),
              "measured_p95_s": reference,
              "p95_relative_error": float(abs(joined.latency_s_predicted.quantile(0.95) / reference - 1))}
    if {"ttft_s_predicted", "ttft_s_measured"}.issubset(joined.columns):
        for suffix in ["predicted", "measured"]:
            values = joined["ttft_s_" + suffix]
            if not np.isfinite(values).all() or (values <= 0).any() or (values > joined["latency_s_" + suffix]).any():
                raise ValueError("Require 0 < TTFT <= end-to-end latency")
        result["ttft_mae_s"] = float(np.abs(joined.ttft_s_predicted - joined.ttft_s_measured).mean())
        result["ttft_p95_relative_error"] = float(abs(joined.ttft_s_predicted.quantile(0.95) / joined.ttft_s_measured.quantile(0.95) - 1))
    joined["latency_error_s"] = error
    return result, joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--predicted-energy", type=Path)
    parser.add_argument("--measured-energy", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/empirical_system_validation.json")
    args = parser.parse_args()
    result, joined = compare_requests(pd.read_csv(args.predicted), pd.read_csv(args.measured))
    if bool(args.predicted_energy) != bool(args.measured_energy):
        parser.error("Supply both energy JSONs, or neither")
    if args.measured_energy:
        expected = json.loads(args.predicted_energy.read_text(encoding="utf-8"))
        observed = json.loads(args.measured_energy.read_text(encoding="utf-8"))
        for key in ["window_start", "window_end", "boundary"]:
            if key not in expected or expected.get(key) != observed.get(key):
                raise ValueError("Energy comparison requires identical windows and boundaries")
        a, b = float(expected["facility_kwh"]), float(observed["facility_kwh"])
        if not np.isfinite(a + b) or a < 0 or b <= 0:
            raise ValueError("Invalid energy values")
        result["facility_energy_relative_error"] = abs(a / b - 1)
    result["experiment_matching_requirement"] = "User must match hardware, model, precision, batching, request lengths, arrival times and measurement boundary"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    joined.to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
