from dataclasses import asdict
from pathlib import Path
import copy
import json
import importlib.metadata
import platform
import threading
import time
import traceback
import numpy as np
import pandas as pd
import psutil
from task4_core.sources import BASE, RELEASE, ROOT, TASK2, read, sha
from task4_core.serialization import digest, lines, save
from task4_core.world import ExogenousWorld
from task4_core.router import PreDirectRouter
from task4_core.window import WindowedGreedy, WindowedSpatial
from task4_core.accounting import evaluate_window
from task4_core.metrics import quality_harms, fairness, failure_decomposition
from task4_core.registry import execution_signature, reusable
from task4_core.validation import validate_execution


def frozen_config(world):
    cfg = {**read(BASE/'configs/core.json')['model'],
           **read(RELEASE.parent/'task_a2/configs/solver_protocol_frozen.json')['model_overrides']}
    cfg.update(controller_w=world.parameters['controller_w'], router_seconds=world.parameters['router_seconds'],
               controller_pue=world.parameters['pue'], export_forecasts=True)
    assert cfg['solver_time_limit'] == 30 and cfg['measured_wall_fraction'] == 0 and cfg['service_guard'] == 1.35
    return cfg


class MemorySampler:
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak, self.samples = self.baseline, 0
        self.stop = threading.Event()
        def observe():
            while not self.stop.is_set():
                self.peak = max(self.peak, self.process.memory_info().rss)
                self.samples += 1
                self.stop.wait(.02)
        self.thread = threading.Thread(target=observe, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


class SimpleRunner:
    def __init__(self, root, router=None):
        self.root = Path(root).resolve()
        if ROOT.resolve() not in self.root.parents:
            raise ValueError('All outputs must stay inside TASK 4')
        self.root.mkdir(parents=True, exist_ok=True)
        self.router = router or PreDirectRouter()
        self._quality = None

    def quality_after_simulation(self, world):
        if self._quality is None:
            wanted = set(pd.read_csv(BASE/'results/data/requests.csv', usecols=['request_id']).request_id)
            outcomes = []
            with (TASK2/'inputs/paired/outcomes.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['request_id'] in wanted:
                        outcomes.append({k: row[k] for k in ['request_id','action_id','quality_true']})
            paired = pd.DataFrame(outcomes).pivot(index='request_id',columns='action_id',values='quality_true')
            self._quality = paired.rename(columns={'S':'q_s','L':'q_l'})
        frame = world.instances[['instance_id','base_request_id']].copy()
        frame = frame.join(self._quality, on='base_request_id', validate='many_to_one')
        return frame.rename(columns={'instance_id':'request_id'})

    def _signature(self, world, jobs, routing, policy, cfg):
        hashes = dict(world.source_hashes)
        hashes.update({str(p): sha(p) for p in sorted((ROOT/'task4_core').glob('*.py'))})
        hashes.update(self.router.hashes)
        hashes.update({str(p): sha(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))})
        return {'source_hashes': hashes, 'routing_hash': digest(routing.to_dict('records')),
                'request_hash': digest([asdict(q) for q in jobs]), 'world_arrays_hash': world.world_id,
                'host_hash': digest(world.hosts), 'network_hash': digest(world.network_rows),
                'carbon_hash': digest(world.carbon_frame.to_dict('records')),
                'forecast_hash': digest(world.forecast_field.to_dict('records')), 'capacity_hash': digest(world.availability_events),
                'horizon_s': 43200., 'anchor_utc': world.anchor_utc, 'controller_config': cfg,
                'solver_protocol': read(RELEASE.parent/'task_a2/configs/solver_protocol_frozen.json'),
                'accounting_version': 'window-v3-causal-hour-field', 'policy': policy,
                'runtime_environment': {'python':platform.python_version(),'platform':platform.platform(),
                    **{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','joblib','lightgbm','threadpoolctl']}}}

    def _append(self, value):
        with (self.root/'ledger.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')

    def execute(self, world, policy, reference=None):
        try:
            return self._execute(world,policy,reference)
        except (PermissionError,KeyboardInterrupt,SystemExit):
            raise
        except Exception as exc:
            folder = self.root/'preflight_failures'/str(time.time_ns())
            folder.mkdir(parents=True)
            record = {'status':'FAILED','world_id':world.world_id,'policy':policy,'folder':str(folder),
                      'stage':'preflight/routing/signature','reference':reference}
            save(folder/'exception.json',{'exception':repr(exc),'traceback':traceback.format_exc()})
            save(folder/'record.json',record)
            self._append({'event':'PREFLIGHT_FAILED',**record})
            return record

    def _execute(self, world, policy, reference=None):
        from experiment import authorize_world
        authorize_world(world, policy)
        if policy not in ['Greedy','Spatial_MILP']:
            raise ValueError('Only final Greedy/Spatial or explicit All-L reference')
        start = time.perf_counter()
        world.verify()
        for path, expected in world.source_hashes.items():
            if sha(path) != expected:
                raise RuntimeError('Source changed after world generation: '+path)
        with MemorySampler() as memory:
            preparation = time.perf_counter()
            jobs, routing = world.routed_requests(self.router, all_l=policy=='AllL_Greedy')
            route_time = time.perf_counter()-preparation
            cfg = frozen_config(world)
            config = self._signature(world, jobs, routing, policy, cfg)
            signature = execution_signature(config)
            world_folder = self.root/'worlds'/world.world_id
            if not world_folder.exists():
                world.save(world_folder)
            else:
                ExogenousWorld.load(world_folder).verify()
            home = self.root/'executions'/signature
            home.mkdir(parents=True, exist_ok=True)
            for previous in sorted(home.glob('attempt_*/record.json')):
                record = read(previous)
                if reusable(record, config):
                    validate_execution(previous.parent, world)
                    self._append({'event':'REUSE_REFERENCE','signature':signature,'world_id':world.world_id,
                                  'reference':reference,'record':str(previous),'new_independent_sample':False})
                    return {**record, 'reused':True}
            index = 1
            while (home/f'attempt_{index:03d}').exists():
                index += 1
            folder = home/f'attempt_{index:03d}'
            folder.mkdir()
            save(folder/'execution.json', {'world_id':world.world_id,'policy':policy,'config':cfg,'signature_config':config,
                                          'reference':reference,'formal_experiment':True, 'study':'simple_assessment_v1_not_C1_C8'})
            routing.to_csv(folder/'routing.csv.gz', index=False, compression='gzip')
            save(folder/'record.json', {'status':'RUNNING','signature':signature,'world_id':world.world_id})
            sim = None
            try:
                import queue_milp
                queue_milp.ROOT = folder/'solver_evidence'
                servers, net, availability, carbon, shocks, forecast = world.environment()
                cls = WindowedSpatial if policy == 'Spatial_MILP' else WindowedGreedy
                began = time.perf_counter()
                sim = cls(jobs,servers,net,availability,carbon,shocks,cfg=cfg,forecast_factory=forecast)
                sim.run()
                simulation_wall = time.perf_counter()-began

                quality = self.quality_after_simulation(world)
                summary, records, ledger = evaluate_window(sim)
                meta = world.instances[['instance_id','base_request_id','prompt_group','eval_name','length_bin_train_only','difficulty_bin_train_only']].rename(
                    columns={'instance_id':'request_id','prompt_group':'group'})
                records = records.merge(meta, on='request_id', validate='one_to_one')
                harms = quality_harms(records,quality)
                for field in ['model_selection_signed_loss','model_selection_harm','delivery_quality_lost',
                              'delivery_incremental_harm','end_to_end_signed_loss','end_to_end_harm',
                              'model_pass_harm','end_to_end_pass_harm','delivery_pass_increment']:
                    summary[field] = float(harms[field].mean())
                summary['pre_realized_gain_bias'] = float(((routing.perturbed_L-routing.perturbed_S)-
                                                          (routing.pre_L-routing.pre_S)).mean())
                summary['pre_rmse_S'] = float(np.sqrt(np.mean(routing.realized_error_S**2)))
                summary['pre_rmse_L'] = float(np.sqrt(np.mean(routing.realized_error_L**2)))
                summary['pre_clip_fraction_S'] = float(routing.clipped_S.mean())
                summary['pre_clip_fraction_L'] = float(routing.clipped_L.mean())
                valid = harms.status.eq('completed') & ~harms.service_constraint_violation
                qm = np.where(harms.selected_model.eq('L'),harms.q_l,harms.q_s)
                summary['effective_delivered_quality_per_input'] = float((valid*qm).mean())
                response = harms.get('completion_s',pd.Series(np.nan,index=harms.index))-harms.arrival_s
                summary['all_input_time_loss'] = float(np.where(harms.status.eq('completed'),
                    np.minimum(response/(harms.deadline_s-harms.arrival_s),2),2).mean())
                failure = failure_decomposition(sim,records,harms)
                summary['observed_mechanism_counts'] = {k:int(failure[k].sum()) for k in
                    ['upstream_model_selection_harm','server_capacity_exposure','queue_expiration',
                     'milp_numerical_rejection','milp_solver_timeout','fallback_used','execution_timeout','window_censoring']}
                summary['primary_service_outcome_counts'] = failure.primary_service_outcome.value_counts().to_dict()
                records.to_csv(folder/'requests.csv.gz',index=False,compression='gzip')
                ledger.to_csv(folder/'energy_ledger.csv.gz',index=False,compression='gzip')
                harms.to_csv(folder/'quality_harms.csv.gz',index=False,compression='gzip')
                failure.to_csv(folder/'failure_causes.csv.gz',index=False,compression='gzip')
                class_weights = pd.read_csv(BASE/'results/data/requests.csv').service_class.value_counts(normalize=True).to_dict()
                disparities = []
                for columns in [['task_type'],['eval_name'],['service_class'],['origin_region'],
                                ['length_bin_train_only'],['difficulty_bin_train_only'],['task_type','service_class']]:
                    grouped = fairness(harms,columns,class_weights)
                    grouped.to_csv(folder/('fairness_'+'_'.join(columns)+'.csv.gz'),index=False,compression='gzip')
                    valid_rates = 1-grouped.violation_rate
                    disparities.append({'groups':columns,'absolute_violation_gap':float(grouped.violation_rate.max()-grouped.violation_rate.min()),
                        'valid_rate_ratio':float(valid_rates.min()/valid_rates.max()) if valid_rates.max()>0 else None,
                        'sparse_prompt_groups_below_30':int((grouped.unique_prompt_groups<30).sum()),
                        'no_demographic_fairness_claim':True})
                save(folder/'fairness_disparities.json',disparities)
                pd.DataFrame(sim.log).to_csv(folder/'events.csv.gz',index=False,compression='gzip')
                pd.DataFrame(sim.controls,columns=['start_s','end_s','region','kind']).to_csv(folder/'controls.csv.gz',index=False,compression='gzip')
                pd.DataFrame(sim.forecast_audit).to_csv(folder/'forecast_audit.csv.gz',index=False,compression='gzip')
                pd.DataFrame(sim.forecasts).to_csv(folder/'issued_forecasts.csv.gz',index=False,compression='gzip')
                lines(folder/'solver_log.jsonl.gz', sim.solver_log)
                save(folder/'future_events_at_H.json',sim.events)
                save(folder/'metrics.json',summary)
                validation = validate_execution(folder,world)
                world.verify()
                save(folder/'validation.json',validation)
                benchmark = {'route_wall_s':route_time,'simulation_wall_s':simulation_wall,
                             'execution_wall_s_before_final_manifest':time.perf_counter()-start,
                             'solver_calls':len(sim.solver_log),'solver_wall_s':summary['solver_wall_seconds'],
                             'sampled_peak_rss_bytes':max(memory.peak,memory.process.memory_info().rss),'rss_baseline_bytes':memory.baseline,'memory_samples':memory.samples,
                             'memory_method':'20ms RSS sampling; process includes persistent input/model caches; not exact transient peak',
                             'artifact_bytes_before_manifest':sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
                             'world_bytes':sum(p.stat().st_size for p in world_folder.rglob('*') if p.is_file()),
                             'requests':len(jobs),'policy':policy,'world_id':world.world_id,'seed':world.seed}
                save(folder/'benchmark.json',benchmark)
                artifacts = {str(p):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='record.json'}
                artifacts.update({str(p):sha(p) for p in sorted(world_folder.rglob('*')) if p.is_file()})
                record = {'status':'COMPLETE_VALIDATED','signature':signature,'world_id':world.world_id,'policy':policy,
                          'folder':str(folder),'artifact_hashes':artifacts,'config':config,'reused':False}
                save(folder/'record.json',record)
                self._append({'event':'COMPLETE_VALIDATED','record':str(folder/'record.json'),'signature':signature,
                              'world_id':world.world_id,'policy':policy,'reference':reference,'new_independent_sample':False})
                return record
            except BaseException as exc:
                if sim is not None:
                    lines(folder/'partial_solver_log.jsonl.gz',sim.solver_log)
                    lines(folder/'partial_events.jsonl.gz',sim.log)
                    save(folder/'partial_records.json',sim.rec)
                    save(folder/'partial_controls.json',sim.controls)
                save(folder/'exception.json',{'exception':repr(exc),'traceback':traceback.format_exc(),
                                             'elapsed_s':time.perf_counter()-start,'metrics_missing_are_not_zero':True})
                record = {'status':'FAILED','signature':signature,'world_id':world.world_id,'policy':policy,'folder':str(folder)}
                save(folder/'record.json',record)
                self._append({'event':'FAILED','record':str(folder/'record.json'),'world_id':world.world_id,'policy':policy})
                if isinstance(exc,(KeyboardInterrupt,SystemExit)):
                    raise
                return record
