from dataclasses import asdict
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT))
from task4_core.world import Scenario

SEEDS = [2029101800,2029101801,2029101802]
POLICIES = ['Greedy','Spatial_MILP']
N = 7335
VERSION = 'Simple-Scenario-Assessment-v1-2026-10-18'
DRY_SEED = 2029101799


def scenarios(n=N):
    return {'Baseline':Scenario(n=n,rho_q=0.),
            'Difficult':Scenario(n=n,hard_fraction=.4,rho_q=0.),
            'Prediction Error':Scenario(n=n,sigma_q=.1,rho_q=0.),
            'Demand Growth':Scenario(n=n,growth=.5,rho_q=0.)}


def authorize_world(world, policy):
    allowed = scenarios(N) if world.seed in SEEDS else scenarios(80) if world.seed==DRY_SEED else {}
    if policy not in POLICIES or world.parameters not in [asdict(s) for s in allowed.values()]:
        raise PermissionError('Only four predeclared simple scenarios and reserved seeds authorized')
    maximum = 11002 if world.seed in SEEDS else 120
    if len(world.instances)>maximum or len(world.hosts)!=6:
        raise PermissionError('No larger worlds or capacity expansion authorized')
    world.verify()


def config():
    return {'version':VERSION,'seeds':SEEDS,'policies':POLICIES,'scenarios':{k:asdict(v) for k,v in scenarios().items()},
            'strategy_execution_limit':24,'horizon_seconds':43200,'growth_rounding':'N + floor(0.5*N) = 11002',
            'prediction_noise':'per prompt-group independent S/L N(0,0.10^2), before clipping; same within duplicates',
            'service_noise':'mean-one lognormal sigma=.15, keyed seed/instance, shared across policies',
            'quality_evaluation':'post-simulation only, pass threshold 0.5',
            'disabled':['C1-C8','CVaR MC','LHS','joint failures','new hosts','months','new optimizer']}
