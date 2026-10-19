from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
import numpy as np
import pandas as pd
from .sources import BASE, RELEASE, ROOT, TASK2, read, sha
from .serialization import digest, save
from .resources import pools_from_hosts
from .window import ENGINE
from src.domain import Request, load_inputs
from src.carbon import Carbon, ForecastView, REGIONS


def rng_for(seed, stream):
    key = hashlib.sha256((str(seed) + '|' + stream).encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:16], 'little'))


@dataclass(frozen=True)
class Scenario:
    n: int = 7335
    hard_fraction: float = None
    compression: float = 1.
    burst_factor: float = 1.
    burst_start: float = 3600.
    burst_duration: float = 900.
    growth: float = 0.
    gain_bias: float = 0.
    quality_bias_s: float = 0.
    quality_bias_l: float = 0.
    sigma_q: float = 0.
    rho_q: float = .5
    sigma_carbon: float = 0.
    rho_carbon: float = .5
    carbon_bias_de: float = 0.
    carbon_bias_fr: float = 0.
    carbon_bias_es: float = 0.
    sigma_service: float = .15
    outage: str = 'none'
    outage_start: float = 3600.
    outage_duration: float = 1800.
    reduced_fraction: float = 0.
    l_hosts: int = 1
    slots_per_host: int = 1
    shared_w: float = 1500.
    gpu_idle_w: float = 175.
    gpu_busy_w: float = 760.
    pue: float = 1.2
    service_scale_s: float = 1.
    service_scale_l: float = 1.
    carbon_scale: float = 1.
    carbon_contrast: float = 1.
    network_delay_scale: float = 1.
    network_energy_scale: float = 1.
    controller_w: float = 10.
    router_seconds: float = .01

    frozen_shock_seed: int = None

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError('Nonfinite scenario: ' + key)
        if not isinstance(self.n, int) or not 1 <= self.n <= 7335:
            raise ValueError('Base cohort size must be 1..7335')
        if self.hard_fraction is not None and not 0 <= self.hard_fraction <= 1:
            raise ValueError('Invalid task mixture')
        for key in ['compression', 'service_scale_s', 'service_scale_l', 'carbon_scale',
                    'network_delay_scale', 'network_energy_scale']:
            if getattr(self, key) <= 0:
                raise ValueError('Nonpositive ' + key)
        if (self.growth < 0 or self.burst_factor < 1 or self.burst_duration <= 0 or self.burst_start < 0
                or self.outage not in {'none', 'FR_L', 'all_L'} or self.outage_start <= 0
                or self.outage_duration <= 0 or not 0 <= self.reduced_fraction <= 1
                or not 0 <= self.rho_q <= 1 or not 0 <= self.rho_carbon <= 1
                or min(self.sigma_q, self.sigma_carbon, self.sigma_service, self.carbon_contrast,
                       self.router_seconds, self.controller_w) < 0
                or not isinstance(self.l_hosts, int) or self.l_hosts < 1
                or not isinstance(self.slots_per_host, int) or not 1 <= self.slots_per_host <= 8):
            raise ValueError('Invalid scenario physical/distribution parameters')


class Availability:

    def __init__(self, events):
        self.events = [tuple(e) for e in events]

    def at(self, region, model, time):
        return [c for t, r, m, c in self.events if (r, m) == (region, model) and t <= time][-1]


class FieldForecast(ForecastView):
    def __init__(self, actual, issued, horizon, field):

        super().__init__(actual, issued, horizon, method='persistence', noise=0.)
        self.grid_hour = int(math.floor(issued / 3600))
        row = field.loc[self.grid_hour]
        self.method = 'persistence_exogenous_hour_field'
        self.table = {(h, r): value * float(row[r]) for (h, r), value in self.table.items()}


@dataclass
class ExogenousWorld:
    world_id: str
    seed: int
    parameters: dict
    anchor_utc: str
    instances: pd.DataFrame
    pre_errors: pd.DataFrame
    forecast_field: pd.DataFrame
    service_shocks: dict
    availability_events: list
    hosts: list
    network_rows: list
    carbon_frame: pd.DataFrame
    carbon_metadata: dict
    source_hashes: dict
    generation_audit: dict

    def payload(self):
        return {**{k: deepcopy(getattr(self, k)) for k in ['seed', 'parameters', 'anchor_utc',
                 'service_shocks', 'availability_events', 'hosts', 'network_rows', 'carbon_metadata',
                 'source_hashes', 'generation_audit']},
                **{k: getattr(self, k).to_dict('records') for k in
                   ['instances', 'pre_errors', 'forecast_field', 'carbon_frame']}}

    def verify(self):
        if digest(self.payload()) != self.world_id:
            raise ValueError('Exogenous world mutated')
        return True

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=False)
        self.verify()
        save(folder / 'world.json', {'world_id': self.world_id, **self.payload()})
        for name in ['instances', 'pre_errors', 'forecast_field', 'carbon_frame']:
            getattr(self, name).to_csv(folder / (name + '.csv.gz'), index=False, compression='gzip')

    @classmethod
    def load(cls, folder):
        value = read(folder / 'world.json')
        for key in ['instances', 'pre_errors', 'forecast_field', 'carbon_frame']:
            value[key] = pd.DataFrame(value[key])
        obj = cls(**value)
        obj.verify()
        return obj

    def environment(self):
        self.verify()
        pools = pools_from_hosts(self.hosts)
        net = {(r['origin'], r['destination']): {k: r[k] for k in ['delay', 'energy', 'allowed']}
               for r in self.network_rows}
        carbon = Carbon(self.carbon_frame, pd.Timestamp(self.anchor_utc), deepcopy(self.carbon_metadata))
        field = self.forecast_field.set_index('grid_hour').copy()
        def forecast(actual, issued, horizon):
            return FieldForecast(actual, issued, horizon, field)
        return pools, net, Availability(self.availability_events), carbon, dict(self.service_shocks), forecast

    def routed_requests(self, router, all_l=False):
        errors = self.pre_errors.set_index('instance_id').loc[self.instances.instance_id]
        inputs = self.instances.rename(columns={'instance_id': 'request_id'})
        route = router.route_frame(inputs, errors.error_s.to_numpy(), errors.error_l.to_numpy())
        if all_l:
            route['selected_model'], route['calls_s'], route['calls_l'] = 'L', 0, 1
            route['fallback_reason'] = 'All_L_reference'
        route['instance_id'] = route.request_id
        mapping = self.instances.set_index('instance_id')
        route['base_request_id'] = route.instance_id.map(mapping.base_request_id)
        route['prompt_group'] = route.instance_id.map(mapping.prompt_group)
        jobs = []
        for row in route.itertuples():
            base = mapping.loc[row.request_id]
            service = base['service_' + row.selected_model.lower()]
            jobs.append(Request(row.request_id, row.selected_model, base.origin_region, row.task_type,
                                base.service_class, float(base.arrival_s), float(base.arrival_s + base.budget_s),
                                float(base.wait_s), float(service), int(base.priority)))
        route['selected_service_s'] = [q.service for q in jobs]
        for m in ['S', 'L']:
            delta = route['perturbed_' + m] - route['pre_' + m]
            route['realized_error_' + m] = delta
            route['clipped_' + m] = route['perturbed_' + m].isin([0., 1.])
        return jobs, route


class ScenarioBuilder:
    def __init__(self):
        self.raw = pd.read_csv(BASE / 'results/data/requests.csv', dtype={'request_id': str})
        self.splits = pd.read_csv(TASK2 / 'runs/experiment/splits.csv', dtype={'request_id': str})
        self.groups = self.splits.set_index('request_id')['group'].to_dict()
        wanted = set(self.raw.request_id)
        minimal = []
        self.train_lengths = []
        train_ids = set(self.splits.loc[self.splits.phase.eq('train'), 'request_id'])
        with (TASK2 / 'inputs/paired/requests.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['request_id'] in wanted:
                    minimal.append({k: row[k] for k in ['request_id', 'prompt', 'eval_name']})
                if row['request_id'] in train_ids:
                    self.train_lengths.append(len(row['prompt']))
        self.prompts = pd.DataFrame(minimal).set_index('request_id')
        self.base_jobs, _, self.base_net, self.anchor = load_inputs()
        self.base_job_map = {q.request_id: q for q in self.base_jobs}
        self.host_frame = pd.read_csv(BASE / 'results/data/physical_host_config.csv')
        carbon = pd.read_csv(BASE / 'results/data/carbon_lifecycle.csv')
        times = pd.to_datetime(carbon.datetime_utc, utc=True)
        self.carbon = carbon[(times >= self.anchor - pd.Timedelta(hours=24)) &
                             (times < self.anchor + pd.Timedelta(hours=13))].copy()
        self.carbon['datetime_utc'] = pd.to_datetime(self.carbon.datetime_utc, utc=True).map(lambda x: x.isoformat())
        self.metadata = read(BASE / 'results/data/carbon_lifecycle.metadata.json')
        paths = list((BASE / 'src').glob('*.py')) + list(RELEASE.glob('*.py')) + list((RELEASE / 'v2').glob('*.py'))
        paths += list((BASE / 'results/data').glob('*.csv')) + list((BASE / 'results/data').glob('*.json'))
        paths += [BASE / 'configs/core.json', BASE.parent / 'data_preparation/config.json',
                  RELEASE.parent / 'task_a2/configs/solver_protocol_frozen.json',
                  TASK2 / 'inputs/paired/requests.jsonl', TASK2 / 'inputs/paired/outcomes.jsonl',
                  TASK2 / 'runs/experiment/splits.csv']
        paths += list(TASK2.glob('*.py'))
        paths += [TASK2 / 'runs/experiment' / p for p in ['pre_full_train.joblib', 'final_model_choice.json',
                  'pre_selection.json', 'final_safety_calibration.json', 'selected_parameters.json', 'cost_predictor.json']]
        self.source_hashes = {str(p): sha(p) for p in sorted(set(paths))}
        self.length_cuts = np.quantile(self.train_lengths, [1/3, 2/3]).tolist()
        prediction_path = TASK2 / 'runs/experiment/pre_predictions.csv.gz'
        predictions = pd.read_csv(prediction_path, usecols=['request_id','split','model_id','A_Baseline'])
        train_s = predictions[predictions.request_id.isin(train_ids) &
                              predictions.model_id.eq('mistralai/mixtral-8x7b-chat')]
        if train_s.empty:
            raise ValueError('Missing train-only difficulty reference')
        self.difficulty_cuts = train_s.A_Baseline.quantile([1/3,2/3]).tolist()
        self.source_hashes[str(prediction_path)] = sha(prediction_path)

    def build(self, scenario, seed):
        scenario.validate()
        p = asdict(scenario)
        base = self.raw.copy()
        if scenario.n < len(base):
            ids = {q.request_id for q in load_inputs(scenario.n)[0]}
            base = base[base.request_id.isin(ids)].copy()
        original = base.copy()


        if scenario.hard_fraction is not None:
            probabilities = np.array([scenario.hard_fraction * 86/1796,
                                      scenario.hard_fraction * 1710/1796, 1 - scenario.hard_fraction])
            expected = probabilities * len(base)
            counts = np.floor(expected).astype(int)
            for j in np.argsort(-(expected-counts), kind='stable')[:len(base)-counts.sum()]:
                counts[j] += 1
            selected = []
            rng = rng_for(seed, 'task_mixture')
            for task, count in zip(['Code', 'Math', 'General'], counts):
                pool = self.raw[self.raw.task_type.eq(task)]
                selected.extend(pool.iloc[rng.integers(0, len(pool), int(count))].to_dict('records'))
            base = pd.DataFrame(selected).iloc[rng.permutation(len(selected))].reset_index(drop=True)
        rows = []
        for j, (_, row) in enumerate(base.iterrows()):
            rid = str(row.request_id)
            q = self.base_job_map[rid]
            arrival = self.base_job_map[str(original.iloc[j].request_id)].arrival
            instance = rid if scenario.hard_fraction is None else f'mix_{j:06d}_{rid}'
            rows.append(self._instance(row, q, instance, arrival, scenario))


        extras = int(math.floor(scenario.growth * scenario.n))
        rng = rng_for(seed, 'growth')
        for j in range(extras):
            source_index, arrival_index = rng.integers(0, len(base), 2)
            row = base.iloc[int(source_index)]
            q = self.base_job_map[str(row.request_id)]
            arrival = self.base_job_map[str(original.iloc[int(arrival_index)].request_id)].arrival
            rows.append(self._instance(row, q, f'growth_{j:06d}_{row.request_id}', arrival, scenario))
        instances = pd.DataFrame(rows)
        support = max(q.arrival for q in self.base_jobs)
        if scenario.burst_factor != 1:
            left, right = scenario.burst_start, scenario.burst_start + scenario.burst_duration
            if not 0 <= left < right <= support:
                raise ValueError('Burst interval outside arrival support')
            weight = support + (scenario.burst_factor-1) * (right-left)
            u = instances.arrival_s.to_numpy() / support * weight
            mapped = np.where(u <= left, u, np.where(u < left+scenario.burst_factor*(right-left),
                               left+(u-left)/scenario.burst_factor, u-(scenario.burst_factor-1)*(right-left)))
            instances['arrival_s'] = mapped
        instances['arrival_s'] /= scenario.compression
        if not instances.arrival_s.between(0, np.nextafter(43200., -np.inf)).all():
            raise ValueError('Arrival outside fixed H')
        instances = instances.sort_values('arrival_s', kind='stable').reset_index(drop=True)
        error_by_group = {}
        for group in sorted(set(instances.prompt_group)):
            u, es, el = rng_for(seed, 'pre|' + group).normal(size=3)
            common = math.sqrt(scenario.rho_q) * u
            independent = math.sqrt(1-scenario.rho_q)
            error_by_group[group] = (scenario.quality_bias_s - scenario.gain_bias/2 + scenario.sigma_q*(common+independent*es),
                                     scenario.quality_bias_l + scenario.gain_bias/2 + scenario.sigma_q*(common+independent*el))
        pre = pd.DataFrame([{'instance_id': r.instance_id, 'prompt_group': r.prompt_group,
                             'error_s': error_by_group[r.prompt_group][0], 'error_l': error_by_group[r.prompt_group][1]}
                            for r in instances.itertuples()])
        forecast = []
        biases = np.array([scenario.carbon_bias_de, scenario.carbon_bias_fr, scenario.carbon_bias_es])
        for hour in range(13):
            z = rng_for(seed, 'carbon|' + str(hour)).normal(size=4)
            errors = biases + scenario.sigma_carbon * (math.sqrt(scenario.rho_carbon)*z[0] + math.sqrt(1-scenario.rho_carbon)*z[1:])
            forecast.append({'grid_hour': hour, **dict(zip(REGIONS, np.exp(errors - scenario.sigma_carbon**2/2)))})
        shocks = {i: float(np.exp(-scenario.sigma_service**2/2 + scenario.sigma_service*rng_for(seed, 'service|' + i).normal()))
                  for i in instances.instance_id}
        hashes = dict(self.source_hashes)
        if scenario.frozen_shock_seed is not None:
            if extras or scenario.hard_fraction is not None or scenario.sigma_service != .15:
                raise ValueError('Frozen shocks only for original equivalence contracts')
            path = RELEASE.parent / 'task_a2/extensions/common_shocks' / ('seed_' + str(scenario.frozen_shock_seed) + '.json')
            full = read(path)
            shocks = {i: float(full[i]) for i in instances.instance_id}
            hashes[str(path)] = sha(path)
        hosts = []
        for h in self.host_frame.itertuples():
            for j in range(scenario.l_hosts if h.model_type == 'L' else 1):
                hosts.append({'host_id': h.host_id if j == 0 else h.host_id + '_extra_' + str(j),
                              'region': h.region, 'model': h.model_type, 'gpus': int(h.gpu_count_per_host),
                              'slots': scenario.slots_per_host, 'shared_w': scenario.shared_w,
                              'gpu_idle_w': scenario.gpu_idle_w, 'gpu_busy_w': scenario.gpu_busy_w,
                              'pue': scenario.pue, 'server_max_w': float(h.server_max_w)})
        pools = pools_from_hosts(hosts)
        events = [(0., r, m, pool['capacity']) for (r, m), pool in pools.items()]
        if scenario.outage != 'none':
            for (r, m), pool in pools.items():
                if m == 'L' and (scenario.outage == 'all_L' or r == 'FR'):
                    events += [(scenario.outage_start, r, m, int(math.floor(pool['capacity']*scenario.reduced_fraction))),
                               (scenario.outage_start+scenario.outage_duration, r, m, pool['capacity'])]
        events.sort()
        net = [{'origin': a, 'destination': b, 'delay': v['delay']*scenario.network_delay_scale,
                'energy': v['energy']*scenario.network_energy_scale, 'allowed': v['allowed']}
               for (a,b), v in self.base_net.items()]
        carbon = self.carbon.copy()
        means = carbon.groupby('datetime_utc').carbon_intensity_gco2e_kwh.transform('mean')
        raw = means + scenario.carbon_contrast*(carbon.carbon_intensity_gco2e_kwh-means)
        carbon['carbon_intensity_gco2e_kwh'] = scenario.carbon_scale*np.maximum(0., raw)
        audit = {'task_counts': instances.task_type.value_counts().to_dict(), 'n': len(instances),
                 'unique_base_ids': instances.base_request_id.nunique(), 'unique_prompt_groups': instances.prompt_group.nunique(),
                 'cloned_quality_is_not_independent': True, 'quality_labels_read_by_generator': False,
                 'carbon_transform_clip_fraction': float((raw < 0).mean()), 'length_cuts_train_only': self.length_cuts,
                 'difficulty_cuts_train_only': self.difficulty_cuts,
                 'streams': 'SHA256(seed|stream); PCG64; no policy or invocation time in keys',
                 'burst_before_compression': True, 'source_service_estimates': 'frozen train-only S/L estimates, already includes L x3'}
        obj = ExogenousWorld('', int(seed), p, self.anchor.isoformat(), instances, pre, pd.DataFrame(forecast), shocks,
                             events, hosts, net, carbon, {**self.metadata, 'hypothetical_transform':
                             scenario.carbon_scale != 1 or scenario.carbon_contrast != 1}, hashes, audit)
        obj.world_id = digest(obj.payload())
        obj.verify()
        return obj

    def _instance(self, row, q, instance, arrival, scenario):
        prompt = self.prompts.loc[str(row.request_id), 'prompt']
        group = hashlib.sha256(' '.join(prompt.split()).encode()).hexdigest()
        expected = self.groups[str(row.request_id)]
        if group != expected:
            raise ValueError('Prompt group provenance mismatch')
        return {'instance_id': instance, 'base_request_id': str(row.request_id), 'prompt_group': group,
                'prompt': prompt, 'eval_name': row.eval_name, 'task_type': row.task_type,
                'origin_region': q.origin, 'service_class': q.service_class, 'arrival_s': float(arrival),
                'budget_s': q.deadline-q.arrival, 'wait_s': q.wait, 'priority': q.priority,
                'service_s': float(row.estimated_service_time_S_seconds * scenario.service_scale_s),
                'service_l': float(row.estimated_service_time_L_seconds * scenario.service_scale_l),
                'length_bin_train_only': int(np.searchsorted(self.length_cuts, len(prompt), side='right')),
                'difficulty_bin_train_only': int(np.searchsorted(self.difficulty_cuts, row.q_hat_S, side='right')),
                'source_phase': 'test'}
