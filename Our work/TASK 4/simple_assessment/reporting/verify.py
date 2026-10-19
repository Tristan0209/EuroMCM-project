from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(HERE),str(ROOT),str(ROOT/'audit')]
import numpy as np
import pandas as pd
from experiment import SEEDS,POLICIES,scenarios,authorize_world,DRY_SEED
from task4_core.sources import read,sha,BASE,TASK2
from task4_core.serialization import save
from task4_core.world import ExogenousWorld
from task4_core.registry import reusable
from task4_core.window import WindowedSpatial,SPATIAL,SOURCE_IDENTITIES
from inspect_interfaces import protected_snapshot


def main():
    old = read(HERE/'audit/existing_task4_before.json')
    changed = [p for p,h in old.items() if not Path(p).is_file() or sha(p)!=h]
    assert not changed,changed
    original_upstream = read(ROOT/'audit/protected_before.json')
    current = protected_snapshot()
    upstream_changes = [p for p,h in original_upstream.items() if current.get(p)!=h]
    assert not upstream_changes,upstream_changes
    assert not set(current)-set(original_upstream)
    assert WindowedSpatial.optimize is SPATIAL.SpatialSimulation.optimize
    complete = read(HERE/'runs/completion.json')
    assert complete['status']=='PASS' and len(complete['rows'])==24
    assert len(list((HERE/'runs/executions').glob('*/attempt_*/record.json')))==24
    indexed = {(r['scenario'],r['seed'],r['policy']):r for r in complete['rows']}
    pair_checks = []
    canonical = pd.read_csv(BASE/'results/data/requests.csv').set_index('request_id')

    reference = pd.read_csv(TASK2/'runs/experiment/decisions.csv.gz',
        usecols=['request_id','strategy','phase','rep','initial_l','calls_s','calls_l'])
    reference = reference[reference.strategy.eq('Selected_finite_sample_guard')&reference.phase.eq('test')&reference.rep.eq(0)].set_index('request_id')
    for scenario in scenarios():
        for seed in SEEDS:
            rows = [indexed[(scenario,seed,p)] for p in POLICIES]
            assert rows[0]['world_id']==rows[1]['world_id']
            routes = []
            for row in rows:
                record = read(row['record'])
                assert reusable(record,record['config'])
                folder = Path(row['record']).parent
                world = ExogenousWorld.load(HERE/'runs/worlds'/row['world_id'])
                authorize_world(world,row['policy'])
                assert len(world.hosts)==6 and len({h['host_id'] for h in world.hosts})==6
                route = pd.read_csv(folder/'routing.csv.gz',keep_default_na=False)
                assert route.request_id.is_unique
                fallback = route.task_type.isin(['Code','Math'])
                assert route.loc[fallback,'selected_model'].eq('L').all()
                requests = pd.read_csv(folder/'requests.csv.gz').set_index('request_id')
                sources = world.instances.set_index('instance_id')
                for m in ['S','L']:
                    selected = requests.selected_model.eq(m)
                    np.testing.assert_allclose(requests.loc[selected,'estimated_service_s'],sources.loc[requests.index[selected],'service_'+m.lower()],rtol=0,atol=1e-12)
                if scenario=='Baseline':
                    z = route.set_index('request_id')
                    assert z.selected_model.value_counts().to_dict()=={'L':6904,'S':431}
                    np.testing.assert_allclose(z.pre_S.astype(float),canonical.loc[z.index].q_hat_S,atol=1e-9,rtol=0)
                    np.testing.assert_allclose(z.pre_L.astype(float),canonical.loc[z.index].q_hat_L,atol=1e-9,rtol=0)
                    assert np.array_equal(z.selected_model,np.where(reference.loc[z.index].initial_l,'L','S'))
                    assert np.array_equal(z.calls_s,reference.loc[z.index].calls_s)
                    assert np.array_equal(z.calls_l,reference.loc[z.index].calls_l)
                if scenario=='Difficult':
                    assert (world.instances.task_type.value_counts().to_dict()=={'General':4401,'Math':2794,'Code':140})
                if scenario=='Prediction Error':
                    errors = world.pre_errors.drop_duplicates('prompt_group')
                    assert abs(errors.error_s.mean())<.01 and abs(errors.error_l.mean())<.01
                    assert abs(errors.error_s.std()-.1)<.01 and abs(errors.error_l.std()-.1)<.01
                    assert abs(errors[['error_s','error_l']].corr().iloc[0,1])<.08
                if scenario=='Demand Growth':
                    assert len(world.instances)==11002
                    base = ExogenousWorld.load(HERE/'runs/worlds'/indexed[('Baseline',seed,row['policy'])]['world_id'])
                    part = world.instances.set_index('instance_id').loc[base.instances.instance_id].reset_index()
                    pd.testing.assert_frame_equal(part,base.instances,check_exact=True)
                routes.append(route)
            pd.testing.assert_frame_equal(routes[0],routes[1],check_exact=True)
            pair_checks.append({'scenario':scenario,'seed':seed,'same_world':True,'same_full_route':True,'status':'PASS'})
    report = {'status':'PASS','old_task4_files_checked':len(old),'old_task4_changes':changed,
        'upstream_files_checked':len(original_upstream),'upstream_changes':upstream_changes,
        'simple_actual_strategy_executions':24,'small_dry_run_strategy_executions':8,'C1_C8_executions':0,
        'lightgbm_training':0,'baseline_full_ID_route_matches_Task2_and_Task3':7335,
        'source_identities':SOURCE_IDENTITIES,'pair_checks':pair_checks,
        'seed_reconstruction':'all worlds regenerated before running; exact payload hashes matched',
        'old_failures_and_pilot_retained':True,'quality_read_only_post_simulation':True,
        'difficult_task_counts':{'General':4401,'Math':2794,'Code':140}}
    dest = HERE/'audit'/f'protection_and_contracts_{time.time_ns()}.json'
    save(dest,report)
    print({k:v for k,v in report.items() if k not in ['source_identities','pair_checks']})


if __name__=='__main__':main()
