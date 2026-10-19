import json
import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, sha256


def check_traces(audit, processed, fingerprint):
    facts = {}
    for name, entry in audit["traces"].items():
        path = Path(entry["source_file"])
        assert path.stat().st_size == entry["source_file_bytes"], "Source size changed"
        assert fingerprint(path) == entry["sha256"], "Source fingerprint changed"
        raw = pd.read_csv(path, nrows=entry["raw_rows_read"])
        clean = pd.read_csv(processed / (name + ".csv"))
        assert len(raw) == entry["raw_rows_read"]
        assert len(clean) == entry["simulation_eligible_rows"]
        assert len(raw) - len(clean) == entry["excluded_rows"]
        assert not clean.request_id.duplicated().any()
        ids = clean.request_id.to_numpy(dtype=int)
        assert np.all((ids >= 0) & (ids < len(raw)))
        assert clean.request_id.eq(ids).all()
        reference = raw.iloc[ids]
        inputs = clean.input_tokens.to_numpy(dtype=float)
        outputs = clean.output_tokens.to_numpy(dtype=float)
        assert np.isfinite(inputs).all() and np.isfinite(outputs).all()
        assert np.all((inputs >= 0) & (inputs % 1 == 0))
        assert np.all((outputs >= 1) & (outputs % 1 == 0))
        np.testing.assert_array_equal(inputs, pd.to_numeric(reference.ContextTokens).to_numpy())
        np.testing.assert_array_equal(outputs, pd.to_numeric(reference.GeneratedTokens).to_numpy())
        timestamps = pd.to_datetime(clean.timestamp, utc=True, errors="raise")
        source_times = pd.to_datetime(reference.TIMESTAMP, utc=True, errors="raise")
        np.testing.assert_array_equal(timestamps.astype("int64"), source_times.astype("int64").to_numpy())
        assert timestamps.is_monotonic_increasing
        expected_arrival = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
        discrepancy = float(np.max(np.abs(clean.arrival_s.to_numpy() - expected_arrival)))
        assert discrepancy < 1e-7
        facts[name] = {"full_file_fingerprint_verified": True,
                       "source_physical_data_rows": entry["source_physical_data_rows"],
                       "rows_checked_against_source": len(clean),
                       "excluded_rows": entry["excluded_rows"],
                       "max_relative_time_error_s": discrepancy,
                       "full_file_field_validation": False}
    return facts


def check_benchmarks(output, fingerprint):
    scope = json.loads((output / "public_benchmark_scope.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(output / "public_measured_benchmarks.csv")
    assert len(frame) == scope["configuration_rows"]
    assert frame[["source_file", "configuration_index"]].duplicated().sum() == 0
    fields = ["median_itl_ms", "p95_itl_ms", "gpu_joules_per_output_token", "gpu_joules_per_request",
              "output_throughput_tokens_per_s", "avg_power_watts_reported", "avg_output_tokens"]
    assert np.isfinite(frame[fields].to_numpy()).all() and (frame[fields] > 0).all().all()
    assert (frame.p95_itl_ms >= frame.median_itl_ms).all()
    assert frame.latency_scope.eq("ITL_not_end_to_end_or_TTFT").all()
    assert frame.measurement_scope.eq("published_GPU_energy_not_complete_server_or_facility").all()
    max_energy_error = 0.0
    max_power_error = 0.0
    single_gpu_rows = 0
    for source in scope["source_files"]:
        path = Path(source)
        assert fingerprint(path) == scope["source_sha256"][source]
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        part = frame[frame.source_file == source]
        assert len(part) == len(raw["configurations"])
        for _, row in part.iterrows():
            reference = raw["configurations"][int(row.configuration_index)]
            assert row.model_id == raw["model_id"]
            for column, key in [("median_itl_ms", "median_itl_ms"), ("p95_itl_ms", "p95_itl_ms"),
                                ("gpu_joules_per_request", "energy_per_request_joules"),
                                ("gpu_joules_per_output_token", "energy_per_token_joules")]:
                assert np.isclose(row[column], reference[key], rtol=1e-10)
            assert reference["p99_itl_ms"] >= reference["p95_itl_ms"]
            energy_error = abs(row.gpu_joules_per_output_token * row.avg_output_tokens / row.gpu_joules_per_request - 1)
            max_energy_error = max(max_energy_error, energy_error)
            assert energy_error < 1e-8
            assert np.isclose(row.gpu_kwh_per_1000_requests, row.gpu_joules_per_request / 3600, rtol=1e-10)
            if row.num_gpus == 1:
                power_error = abs(row.gpu_joules_per_output_token * row.output_throughput_tokens_per_s / row.avg_power_watts_reported - 1)
                max_power_error = max(max_power_error, power_error)
                single_gpu_rows += 1
                assert power_error < 1e-8
    return {"configurations_checked": len(frame), "models": frame.model_id.nunique(),
            "single_gpu_configurations_checked": single_gpu_rows,
            "max_energy_field_identity_relative_error": max_energy_error,
            "max_single_gpu_power_identity_relative_error": max_power_error,
            "interpretation": "source transcription and dimensional identities; not independent physical validation"}


def check_carbon(audit, processed, fingerprint):
    factors = pd.read_csv(processed / "annual_carbon_factors.csv")
    assert not factors.scenario.duplicated().any()
    assert np.isfinite(factors.intensity).all() and (factors.intensity >= 0).all()
    assert factors.unit.eq("gCO2e/kWh").all()
    for source, rows in factors.groupby("source"):
        assert rows.source_sha256.nunique() == 1
        path = Path(source)
        assert fingerprint(path) == rows.source_sha256.iloc[0]
        if path.name == "greenhouse_gas_emission_intensity_of_electricity_generation.csv":
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            header = next(index for index, line in enumerate(lines) if line.startswith("Year,"))
            original = pd.read_csv(io.StringIO("\n".join(lines[header:])))
            field = "Greenhouse gas (GHG) emission intensity (gCO2e/kWh)"
            original[field] = pd.to_numeric(original[field], errors="coerce")
            original = original.dropna(subset=[field])
            assert len(rows) == len(original)
            for _, row in rows.iterrows():
                reference = original[original.Year == row.year]
                assert len(reference) == 1 and row.region == "EU27"
                assert np.isclose(row.intensity, reference[field].iloc[0], rtol=1e-12)
        else:
            columns = ["Area", "Year", "Electricity source", "Emissions intensity (gCO2e/kWh)"]
            selected = []
            for chunk in pd.read_csv(path, usecols=columns, chunksize=100000):
                selected.append(chunk[(chunk.Year.isin(rows.year)) & chunk.Area.isin(rows.region)
                                      & (chunk["Electricity source"] == "Total generation")])
            original = pd.concat(selected, ignore_index=True)
            assert len(original) == len(rows)
            for _, row in rows.iterrows():
                reference = original[(original.Year == row.year) & (original.Area == row.region)]
                assert len(reference) == 1
                assert np.isclose(row.intensity, reference["Emissions intensity (gCO2e/kWh)"].iloc[0], rtol=1e-12)
    gb_info = audit["neso"]
    assert fingerprint(Path(gb_info["source_file"])) == gb_info["sha256"]
    raw = json.loads(Path(gb_info["source_file"]).read_text(encoding="utf-8-sig"))
    gb = pd.read_csv(processed / "gb_carbon_intervals.csv")
    starts = pd.to_datetime(gb.start_utc, utc=True)
    ends = pd.to_datetime(gb.end_utc, utc=True)
    origin = pd.Timestamp("2024-05-12T00:00:00Z")
    source_records = {r["from"]: r for r in raw["data"] if pd.Timestamp(r["to"]) > origin}
    assert len(gb) == len(source_records) == gb_info["usable_intervals"]
    for index, row in gb.iterrows():
        reference = source_records[starts.iloc[index].strftime("%Y-%m-%dT%H:%MZ")]
        assert reference["intensity"]["actual"] is not None
        assert row.intensity == reference["intensity"]["actual"]
        assert ends.iloc[index] == pd.Timestamp(reference["to"])
    assert gb.start_s.iloc[0] == 0
    assert np.isfinite(gb.intensity).all() and (gb.intensity >= 0).all()
    assert (gb.end_s > gb.start_s).all()
    np.testing.assert_allclose(gb.start_s.to_numpy()[1:], gb.end_s.to_numpy()[:-1], atol=1e-7, rtol=0)
    np.testing.assert_allclose(gb.start_s, np.maximum(0, (starts - origin).dt.total_seconds()), atol=1e-7, rtol=0)
    np.testing.assert_allclose(gb.end_s, (ends - origin).dt.total_seconds(), atol=1e-7, rtol=0)
    assert gb.unit.eq("gCO2/kWh").all()
    return {"annual_factor_rows": len(factors), "annual_factor_values_rechecked_against_source": True,
            "neso_intervals": len(gb),
            "neso_coverage_seconds": float(gb.end_s.iloc[-1]),
            "missing_actual_values": int(gb.intensity.isna().sum()),
            "gaps_or_overlaps": 0, "different_CO2_CO2e_boundaries_retained": True}


def main():
    processed, output = ROOT / "data/processed", ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((processed / "data_audit.json").read_text(encoding="utf-8"))
    fingerprints = {}

    def fingerprint(path):
        key = str(path.resolve())
        if key not in fingerprints:
            fingerprints[key] = sha256(path)
        return fingerprints[key]

    checks = {}
    for name, function in [("azure_sources_and_preprocessing", lambda: check_traces(audit, processed, fingerprint)),
                           ("public_measurement_records", lambda: check_benchmarks(output, fingerprint)),
                           ("carbon_records_and_coverage", lambda: check_carbon(audit, processed, fingerprint))]:
        try:
            checks[name] = {"passed": True, "details": function()}
        except Exception as error:
            checks[name] = {"passed": False, "error": str(error), "error_type": type(error).__name__}
    passed = all(item["passed"] for item in checks.values())
    report = {"passed": passed, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "checks": checks, "source_files_fingerprinted": len(fingerprints),
              "scope": "complete-file fingerprints; parsed prefix and published-field consistency; not all full-trace rows or physical truth"}
    (output / "data_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
