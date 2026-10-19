import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd


def features(input_tokens, output_tokens):
    inputs = np.asarray(input_tokens, dtype=float)
    outputs = np.asarray(output_tokens, dtype=float)
    if np.any(~np.isfinite(inputs)) or np.any(inputs < 0):
        raise ValueError("Input lengths must be finite and nonnegative")
    if np.any(~np.isfinite(outputs)) or np.any(outputs < 1):
        raise ValueError("Simulation requires at least one output token")
    if np.any(inputs % 1 != 0) or np.any(outputs % 1 != 0):
        raise ValueError("Token lengths must be integers")
    x = inputs / 1000.0
    y = (outputs - 1.0) / 1000.0
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])


def predict_service(requests, profile, seed=0, stochastic=True):
    matrix = features(requests["input_tokens"], requests["output_tokens"])
    pre_coef = np.asarray(profile["prefill_coefficients"], dtype=float)
    dec_coef = np.asarray(profile["decode_coefficients"], dtype=float)
    if pre_coef.shape != (6,) or dec_coef.shape != (6,) or np.any(pre_coef < 0) or np.any(dec_coef < 0):
        raise ValueError("Each profile needs six finite nonnegative coefficients")
    first = matrix @ pre_coef
    decode = matrix @ dec_coef
    if not np.all(np.isfinite(first + decode)) or np.any(first <= 0) or np.any(decode < 0):
        raise ValueError("Service-time predictions must be positive and finite")
    if stochastic:
        sigma = float(profile.get("residual_log_sigma", 0.0))
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Residual log sigma must be finite and nonnegative")
        rng = np.random.default_rng(seed)

        multiplier = np.exp(rng.normal(-0.5 * sigma * sigma, sigma, len(first)))
        first = first * multiplier
        decode = decode * multiplier
    return first, first + decode


def schedule(arrivals, service_s, first_service_s, replicas=1, overhead_s=0.0):
    arrivals = np.asarray(arrivals, dtype=float)
    service = np.asarray(service_s, dtype=float)
    first = np.asarray(first_service_s, dtype=float)
    if not len(arrivals) or arrivals.shape != service.shape or arrivals.shape != first.shape:
        raise ValueError("Nonempty arrays with matching shapes are required")
    if replicas < 1 or int(replicas) != replicas:
        raise ValueError("replicas must be a positive integer")
    if np.any(~np.isfinite(arrivals)) or np.any(arrivals < 0) or np.any(np.diff(arrivals) < 0):
        raise ValueError("Arrival times must be finite, sorted and nonnegative")
    if np.any(~np.isfinite(service)) or np.any(service <= 0) or np.any(~np.isfinite(first)) or np.any(first <= 0) or np.any(first > service):
        raise ValueError("Require 0 < first-token service <= full service")
    if not np.isfinite(overhead_s) or overhead_s < 0:
        raise ValueError("Overhead must be finite and nonnegative")
    pool = [(0.0, replica) for replica in range(int(replicas))]
    heapq.heapify(pool)
    starts = np.empty(len(arrivals))
    finishes = np.empty(len(arrivals))
    servers = np.empty(len(arrivals), dtype=int)
    for index, (arrival, duration) in enumerate(zip(arrivals, service)):
        available, replica = heapq.heappop(pool)
        start = max(float(arrival), available)
        finish = start + float(duration)
        starts[index], finishes[index], servers[index] = start, finish, replica
        heapq.heappush(pool, (finish, replica))
    wait = starts - arrivals
    return pd.DataFrame({
        "arrival_s": arrivals, "start_s": starts, "finish_s": finishes,
        "replica": servers, "queue_wait_s": wait, "service_s": service,
        "ttft_s": overhead_s + wait + first,
        "latency_s": overhead_s + finishes - arrivals,
    })


@dataclass
class CarbonSeries:
    starts: np.ndarray
    ends: np.ndarray
    values: np.ndarray
    source: str
    unit: str
    boundary: str

    def __post_init__(self):
        self.starts = np.asarray(self.starts, dtype=float)
        self.ends = np.asarray(self.ends, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if not len(self.starts) or self.starts.shape != self.ends.shape or self.starts.shape != self.values.shape:
            raise ValueError("Carbon intervals must be nonempty and have matching shapes")
        if np.any(~np.isfinite(self.starts)) or np.any(~np.isfinite(self.ends)) or np.any(~np.isfinite(self.values)):
            raise ValueError("Carbon series must contain finite values")
        if np.any(self.ends <= self.starts) or np.any(self.values < 0):
            raise ValueError("Invalid carbon interval or intensity")
        if len(self.starts) > 1 and not np.allclose(self.starts[1:], self.ends[:-1], atol=1e-7, rtol=0):
            raise ValueError("Carbon intervals must be sorted, contiguous and non-overlapping")

    def at(self, times):
        times = np.asarray(times, dtype=float)
        indices = np.searchsorted(self.starts, times, side="right") - 1
        if np.any(indices < 0) or np.any(indices >= len(self.starts)):
            raise ValueError("Carbon series does not cover experiment start")
        if np.any(times >= self.ends[indices]):
            raise ValueError("Carbon series does not cover experiment end; no silent extrapolation")
        return self.values[indices]


def integrate_energy(trace, replicas, idle_w, busy_w, pue, carbon, observation_end_s=None, *, shared_w=None, gpu_count=8, server_max_w=None):





    if replicas < 1 or int(replicas) != replicas or not np.isfinite(idle_w + busy_w + pue) or idle_w < 0 or busy_w < idle_w or pue < 1:
        raise ValueError("Require replicas>=1, 0<=idle<=busy and PUE>=1")
    last_arrival = float(trace["arrival_s"].max())
    if observation_end_s is None:
        observation_end_s = last_arrival
    if observation_end_s < last_arrival or not np.isfinite(observation_end_s):
        raise ValueError("Observation end must include all submitted requests")
    horizon = max(float(observation_end_s), float(trace["finish_s"].max()))
    starts = trace["start_s"].to_numpy()
    finishes = trace["finish_s"].to_numpy()
    grid_boundaries = np.r_[carbon.starts, carbon.ends]
    grid_boundaries = grid_boundaries[(grid_boundaries > 0) & (grid_boundaries < horizon)]
    times = np.unique(np.r_[0.0, horizon, starts, finishes, grid_boundaries])
    changes = np.zeros(len(times), dtype=int)
    np.add.at(changes, np.searchsorted(times, starts), 1)
    np.add.at(changes, np.searchsorted(times, finishes), -1)
    busy_count = np.cumsum(changes)[:-1]
    if np.any(busy_count < 0) or np.any(busy_count > replicas):
        raise ValueError("Busy-server count violates replica capacity")
    intervals = np.diff(times)



    if shared_w is not None:
        if not np.isfinite(shared_w) or shared_w < 0 or not isinstance(gpu_count, int) or gpu_count < replicas:
            raise ValueError("Invalid shared power or physical GPU count")
        baseline_power = shared_w + gpu_count * idle_w
    else:
        baseline_power = replicas * idle_w
    power = baseline_power + (busy_w - idle_w) * busy_count
    if server_max_w is not None and (not np.isfinite(server_max_w) or np.max(power) > server_max_w + 1e-8):
        raise ValueError("Modeled server power exceeds specified maximum")
    intensity = carbon.at(times[:-1] + 0.5 * intervals)
    it_kwh = power * intervals / 3.6e6
    facility_kwh = it_kwh * pue
    emissions = facility_kwh * intensity
    expected_joules = baseline_power * horizon + (busy_w - idle_w) * float(trace["service_s"].sum())
    if not np.isclose(it_kwh.sum() * 3.6e6, expected_joules, rtol=1e-9, atol=1e-6):
        raise ArithmeticError("Energy conservation check failed")
    timeline = pd.DataFrame({
        "start_s": times[:-1], "end_s": times[1:], "busy_replicas": busy_count,
        "it_power_w": power, "intensity": intensity,
        "it_kwh": it_kwh, "facility_kwh": facility_kwh, "emissions_g": emissions,
    })
    total = {
        "horizon_including_drain_s": horizon,
        "drain_s": max(0.0, horizon - observation_end_s),
        "it_kwh": float(it_kwh.sum()), "facility_kwh": float(facility_kwh.sum()),
        "emissions_g": float(emissions.sum()), "carbon_unit": carbon.unit,
        "carbon_boundary": carbon.boundary, "carbon_source": carbon.source,
        "utilization_observed": float(trace["service_s"].sum()) / (replicas * horizon),
    }
    return total, timeline


def metrics(trace, energy, deadline_s):
    count = len(trace)
    if deadline_s <= 0:
        raise ValueError("Deadline must be positive")
    return {
        "requests": count, "mean_latency_s": float(trace.latency_s.mean()),
        "p50_latency_s": float(trace.latency_s.quantile(0.5)),
        "p95_latency_s": float(trace.latency_s.quantile(0.95)),
        "p99_latency_s": float(trace.latency_s.quantile(0.99)),
        "mean_ttft_s": float(trace.ttft_s.mean()),
        "p95_ttft_s": float(trace.ttft_s.quantile(0.95)),
        "mean_queue_wait_s": float(trace.queue_wait_s.mean()),
        "late_fraction": float((trace.latency_s > deadline_s).mean()),
        "deadline_s": float(deadline_s),
        "facility_kwh_per_1000_requests": energy["facility_kwh"] / count * 1000,
        "emissions_g_per_1000_requests": energy["emissions_g"] / count * 1000,
        **energy,
    }
