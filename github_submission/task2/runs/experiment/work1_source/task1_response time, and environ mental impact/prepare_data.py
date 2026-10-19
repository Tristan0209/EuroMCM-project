import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, DATASETS, sha256


def audit_trace_file(path):





    import hashlib
    digest = hashlib.sha256()
    newlines, last_byte, size = 0, b"", 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
            newlines += block.count(b"\n")
            last_byte = block[-1:]
            size += len(block)
    lines = newlines + int(bool(size) and last_byte != b"\n")
    return {"source_file_bytes": size, "source_physical_data_rows": max(0, lines - 1),
            "sha256": digest.hexdigest(),
            "full_file_audit": "SHA256 and physical-line count; parsed validity is checked only for rows_read"}


def prepare_trace(path, name, output, max_requests):
    raw = pd.read_csv(path, nrows=max_requests)
    required = ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
    if not set(required).issubset(raw.columns):
        raise ValueError("Missing Azure columns: %s" % path)
    timestamp = pd.to_datetime(raw.TIMESTAMP, utc=True, errors="coerce")
    inputs = pd.to_numeric(raw.ContextTokens, errors="coerce")
    outputs = pd.to_numeric(raw.GeneratedTokens, errors="coerce")
    valid = timestamp.notna() & inputs.notna() & outputs.notna()
    valid &= np.isfinite(inputs) & np.isfinite(outputs)
    valid &= (inputs >= 0) & (outputs >= 1) & (inputs % 1 == 0) & (outputs % 1 == 0)
    reasons = {
        "invalid_timestamp": int(timestamp.isna().sum()),
        "invalid_input_length": int((~np.isfinite(inputs) | (inputs < 0) | (inputs % 1 != 0)).sum()),
        "invalid_output_length": int((~np.isfinite(outputs) | (outputs < 1) | (outputs % 1 != 0)).sum()),
    }
    data = pd.DataFrame({"request_id": np.arange(len(raw)), "timestamp": timestamp,
                         "input_tokens": inputs, "output_tokens": outputs})
    rejected = data.loc[~valid].copy()
    if len(rejected):
        rejected.to_csv(output / (name + "_excluded.csv"), index=False, encoding="utf-8")
    data = data.loc[valid].sort_values("timestamp", kind="stable").reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("At least two valid arrivals are required")
    origin = data.timestamp.iloc[0]
    data["arrival_s"] = (data.timestamp - origin).dt.total_seconds()
    data["input_tokens"] = data.input_tokens.astype(int)
    data["output_tokens"] = data.output_tokens.astype(int)
    data.to_csv(output / (name + ".csv"), index=False, encoding="utf-8")
    return {
        "source_file": str(path.resolve()), **audit_trace_file(path),
        "raw_rows_read": len(raw), "simulation_eligible_rows": len(data),
        "excluded_rows": len(rejected), "original_start_utc": origin.isoformat(),
        "invalid_reason_counts_may_overlap": reasons,
        "original_span_s": float(data.arrival_s.iloc[-1]),
        "observed_prefix_rate_per_s": (len(data) - 1) / max(float(data.arrival_s.iloc[-1]), 1e-9),
        "duplicate_timestamps_retained": int(data.timestamp.duplicated().sum()),
        "has_measured_latency": False, "has_measured_energy": False,
        "sample_design": "contiguous prefix, not representative of a full week",
    }


def eea_eu_factors(path):
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Year,")), None)
    if start is None:
        raise ValueError("The selected EEA file is not the EU year/intensity series")
    frame = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    column = "Greenhouse gas (GHG) emission intensity (gCO2e/kWh)"
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    records = []
    for _, row in frame.dropna(subset=[column]).iterrows():
        year = int(row.Year)
        records.append({"scenario": "EEA_EU27_%d" % year, "region": "EU27", "year": year,
                        "intensity": float(row[column]), "unit": "gCO2e/kWh",
                        "boundary": "annual_production_ghg_EEA_not_marginal",
                        "source": str(path), "source_sha256": sha256(path)})
    return records


def ember_factors(path, year=2024):


    records = []
    columns = ["Area", "Year", "Electricity source", "Emissions intensity (gCO2e/kWh)"]
    source_hash = sha256(path)
    regions = {"France", "Germany", "Poland", "Sweden"}
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100000):
        rows = chunk[(chunk.Year == year) & (chunk["Electricity source"] == "Total generation") & chunk.Area.isin(regions)]
        for _, row in rows.iterrows():
            value = float(row["Emissions intensity (gCO2e/kWh)"])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid Ember emission factor")
            records.append({"scenario": "Ember_%s_%d" % (row.Area, year), "region": row.Area,
                            "year": year, "intensity": value, "unit": "gCO2e/kWh",
                            "boundary": "annual_production_electricity_lifecycle_Ember_not_marginal",
                            "source": str(path), "source_sha256": source_hash})
    if len(records) != len(regions):
        raise ValueError("Expected one 2024 Total generation factor for each selected Ember country")
    return records


def gb_intervals(path):
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    origin = pd.Timestamp("2024-05-12T00:00:00Z")
    records = []
    for row in raw["data"]:
        start = pd.Timestamp(row["from"])
        end = pd.Timestamp(row["to"])
        if end <= origin:
            continue
        if end <= start:
            raise ValueError("NESO interval must have positive duration")
        value = row["intensity"].get("actual")
        if value is None:
            raise ValueError("Missing NESO actual intensity; do not silently substitute a forecast")
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError("NESO actual intensity must be finite and nonnegative")
        records.append({"start_s": max(0.0, (start - origin).total_seconds()),
                        "end_s": (end - origin).total_seconds(), "intensity": float(value),
                        "start_utc": start.isoformat(), "end_utc": end.isoformat(),
                        "unit": "gCO2/kWh", "source": "NESO_actual_GB",
                        "boundary": "GB_operational_CO2_official_estimate_not_CO2e_or_marginal"})
    result = pd.DataFrame(records).sort_values("start_s").reset_index(drop=True)
    if result.empty or result.start_s.iloc[0] != 0:
        raise ValueError("NESO history does not cover the chosen experiment start")
    if len(result) > 1 and not np.allclose(result.start_s.to_numpy()[1:], result.end_s.to_numpy()[:-1], atol=1e-7, rtol=0):
        raise ValueError("NESO history contains missing or overlapping intervals")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-data", type=Path, default=DATASETS)
    parser.add_argument("--max-requests", type=int, default=20000)
    args = parser.parse_args()
    output = ROOT / "data/processed"
    if args.max_requests < 2:
        parser.error("--max-requests must be at least 2")
    output.mkdir(parents=True, exist_ok=True)
    audit = {"traces": {}, "warnings": []}
    for name in ["azure_conversation", "azure_code"]:
        downloaded_name = "AzureLLMInferenceTrace_conv_1week.csv" if name == "azure_conversation" else "AzureLLMInferenceTrace_code_1week.csv"
        path = args.existing_data / "Azure" / downloaded_name
        audit["traces"][name] = prepare_trace(path, name, output, args.max_requests)
    factors = []
    eea_paths = [p for p in args.existing_data.rglob("*.csv") if p.name == "greenhouse_gas_emission_intensity_of_electricity_generation.csv"]
    if len(eea_paths) == 1:
        factors.extend(eea_eu_factors(eea_paths[0]))
    else:
        audit["warnings"].append("EEA EU series unavailable; obtain it from the source page")
    ember = args.existing_data / "Ember/release_generation_yearly_lower.csv"
    if ember.exists():
        factors.extend(ember_factors(ember))
    else:
        audit["warnings"].append("Existing Ember yearly CSV unavailable")
    if not factors:
        raise FileNotFoundError("No actual annual carbon factors found; see DATA_SOURCES.md")
    pd.DataFrame(factors).to_csv(output / "annual_carbon_factors.csv", index=False, encoding="utf-8")
    gb_path = args.existing_data / "NESO" / "NESO.JSON"
    gb = gb_intervals(gb_path)
    gb.to_csv(output / "gb_carbon_intervals.csv", index=False, encoding="utf-8")
    audit["neso"] = {"source_file": str(gb_path.resolve()), "sha256": sha256(gb_path),
                     "usable_intervals": len(gb), "coverage_start_s": float(gb.start_s.min()),
                     "coverage_end_s": float(gb.end_s.max())}
    audit["carbon_sources_are_not_interchangeable"] = "EEA CO2e, Ember lifecycle electricity CO2e and NESO operational CO2 have different boundaries"
    audit["eea_country_file"] = "Not used: the existing export contains conflicting packed trailing columns"
    (output / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
