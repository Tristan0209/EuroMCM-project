import json
import math
import unittest

import numpy as np

from core import schedule
from common import ROOT


def main():
    suite = unittest.defaultTestLoader.discover(start_dir=str(ROOT),pattern="test_*.py",top_level_dir=str(ROOT),)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    rng = np.random.default_rng(20261016)
    size, warmup = 300000, 10000
    arrivals = np.cumsum(rng.exponential(1 / 0.3, size))
    service = rng.exponential(2.0, size)
    trace = schedule(arrivals, service, service, 1, 0.05).iloc[warmup:]
    expected_mean = 1 / (0.5 - 0.3) + 0.05
    expected_p95 = -math.log(0.05) / (0.5 - 0.3) + 0.05
    actual_mean, actual_p95 = float(trace.latency_s.mean()), float(trace.latency_s.quantile(0.95))
    report = {"kind": "synthetic_implementation_check_not_empirical_validation",
              "tests_passed": result.testsRun, "requests_after_warmup": len(trace),
              "simulated_mean_s": actual_mean, "analytical_mean_s": expected_mean,
              "simulated_p95_s": actual_p95, "analytical_p95_s": expected_p95,
              "mean_relative_error": abs(actual_mean / expected_mean - 1),
              "p95_relative_error": abs(actual_p95 / expected_p95 - 1)}
    if max(report["mean_relative_error"], report["p95_relative_error"]) > 0.04:
        raise AssertionError("M/M/1 verification differs from theory by more than 4%")
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    (output / "implementation_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
