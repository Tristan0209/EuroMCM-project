import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from core import CarbonSeries, integrate_energy, predict_service, schedule
from prepare_data import audit_trace_file, prepare_trace, gb_intervals, eea_eu_factors
from validate_system import compare_requests


class QueueAndEnergyTests(unittest.TestCase):
    def test_fcfs_and_capacity(self):
        trace = schedule([0, 0, 0], [2, 4, 1], [1, 1, 1], replicas=2)
        np.testing.assert_allclose(trace.start_s, [0, 0, 2])
        np.testing.assert_allclose(trace.finish_s, [2, 4, 3])
        self.assertTrue((trace.ttft_s <= trace.latency_s).all())

    def test_exact_piecewise_energy(self):
        trace = schedule([0, 3], [1, 1], [0.5, 0.5])
        grid = CarbonSeries([0, 2], [2, 5], [100, 200], "test", "gCO2e/kWh", "test")
        energy, _ = integrate_energy(trace, 1, 10, 30, 1.2, grid, 5)
        self.assertAlmostEqual(energy["it_kwh"] * 3.6e6, 90)
        self.assertAlmostEqual(energy["emissions_g"], 14000 / 3.6e6 * 1.2)

    def test_multiple_replica_idle_energy_and_drain(self):
        trace = schedule([0, 1], [10, 2], [0.5, 0.5], replicas=2)
        grid = CarbonSeries([0], [100], [100], "test", "gCO2e/kWh", "test")
        energy, _ = integrate_energy(trace, 2, 10, 30, 1, grid, 2)
        self.assertAlmostEqual(energy["it_kwh"] * 3.6e6, 2 * 10 * 10 + 20 * 12)
        self.assertEqual(energy["drain_s"], 8)

    def test_dgx_shared_power_and_idle_gpus(self):
        trace = schedule([0, 1], [2, 1], [1, 1], replicas=2)
        carbon = CarbonSeries([0], [5], [200], "test", "gCO2e/kWh", "test")
        result, timeline = integrate_energy(trace, 2, 175, 760, 1.2, carbon, 4,
                                            shared_w=1500, gpu_count=8, server_max_w=14300)
        expected = (1500 + 8 * 175) * 4 + (760 - 175) * 3
        self.assertAlmostEqual(result["it_kwh"] * 3.6e6, expected)
        self.assertAlmostEqual(result["emissions_g"], result["facility_kwh"] * 200)
        self.assertTrue((timeline.it_power_w <= 14300).all())

    def test_carbon_gap_and_extrapolation_rejected(self):
        with self.assertRaises(ValueError):
            CarbonSeries([0, 3], [2, 5], [100, 200], "test", "g/kWh", "test")
        grid = CarbonSeries([0], [1], [100], "test", "g/kWh", "test")
        trace = schedule([0], [2], [1])
        with self.assertRaises(ValueError):
            integrate_energy(trace, 1, 10, 30, 1, grid)

    def test_invalid_service_or_arrivals_rejected(self):
        with self.assertRaises(ValueError):
            schedule([1, 0], [1, 1], [0.5, 0.5])
        with self.assertRaises(ValueError):
            schedule([0], [1], [2])

    def test_single_output_token_has_no_decode(self):
        requests = pd.DataFrame({"input_tokens": [100], "output_tokens": [1]})
        profile = {"prefill_coefficients": [1, 0, 0, 0, 0, 0],
                   "decode_coefficients": [0, 0, 8, 0, 1, 1], "residual_log_sigma": 0}
        first, total = predict_service(requests, profile)
        np.testing.assert_equal(first, total)

    def test_same_seed_is_paired_across_configurations(self):
        requests = pd.DataFrame({"input_tokens": [100, 200], "output_tokens": [10, 20]})
        profile = {"prefill_coefficients": [1, 0, 0, 0, 0, 0],
                   "decode_coefficients": [0, 0, 8, 0, 0, 0], "residual_log_sigma": 0.2}
        a = predict_service(requests, profile, seed=7)[1]
        b = predict_service(requests, profile, seed=7)[1]
        np.testing.assert_equal(a, b)


class DataAndValidationTests(unittest.TestCase):
    def test_validation_requires_exact_request_sets(self):
        a = pd.DataFrame({"request_id": [1, 2], "latency_s": [1.0, 2.0]})
        b = pd.DataFrame({"request_id": [1], "latency_s": [1.0]})
        with self.assertRaises(ValueError):
            compare_requests(a, b)
        report, _ = compare_requests(a, a)
        self.assertEqual(report["latency_mae_s"], 0)

    def test_validation_rejects_duplicate_ids_and_impossible_ttft(self):
        frame = pd.DataFrame({"request_id": [1, 2], "latency_s": [1.0, 2.0], "ttft_s": [0.5, 3.0]})
        with self.assertRaises(ValueError):
            compare_requests(frame, frame)
        duplicate = frame.assign(request_id=[1, 1])
        with self.assertRaises(ValueError):
            compare_requests(duplicate, duplicate)

    def test_trace_cleaning_retains_valid_simultaneous_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame({
                "TIMESTAMP": ["2024-05-12T00:00:02Z", "2024-05-12T00:00:00Z",
                              "2024-05-12T00:00:00Z", "bad", "2024-05-12T00:00:03Z",
                              "2024-05-12T00:00:04Z"],
                "ContextTokens": [20, 0, 30, 10, -1, 1.5],
                "GeneratedTokens": [2, 1, 4, 2, 0, 2]}).to_csv(source, index=False)
            report = prepare_trace(source, "trace", root, 6)
            clean = pd.read_csv(root / "trace.csv")
            self.assertEqual(report["excluded_rows"], 3)
            self.assertEqual(report["duplicate_timestamps_retained"], 1)
            self.assertEqual(clean.request_id.tolist(), [1, 2, 0])
            self.assertEqual(clean.arrival_s.tolist(), [0, 0, 2])
            self.assertEqual(len(pd.read_csv(root / "trace_excluded.csv")), 3)

    def test_full_file_audit_handles_crcrlf_and_final_line(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            content = b"a,b\r\r\n1,2\r\r\n3,4"
            source.write_bytes(content)
            report = audit_trace_file(source)
            self.assertEqual(report["source_physical_data_rows"], 2)
            self.assertEqual(report["sha256"], hashlib.sha256(content).hexdigest())

    def test_neso_rejects_missing_actual_and_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "neso.json"
            records = [{"from": "2024-05-12T00:00Z", "to": "2024-05-12T00:30Z",
                        "intensity": {"actual": None, "forecast": 100}}]
            source.write_text(json.dumps({"data": records}), encoding="utf-8")
            with self.assertRaises(ValueError):
                gb_intervals(source)
            records[0]["intensity"]["actual"] = 100
            records.append({"from": "2024-05-12T01:00Z", "to": "2024-05-12T01:30Z",
                            "intensity": {"actual": 100}})
            source.write_text(json.dumps({"data": records}), encoding="utf-8")
            with self.assertRaises(ValueError):
                gb_intervals(source)

    def test_eea_targets_are_not_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "eea.csv"
            source.write_text('Metadata\nYear,"Greenhouse gas (GHG) emission intensity (gCO2e/kWh)",Target\n2024,183.44,\n2030,,100\n', encoding="utf-8")
            records = eea_eu_factors(source)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["year"], 2024)
            self.assertEqual(records[0]["intensity"], 183.44)


if __name__ == "__main__":
    unittest.main()
