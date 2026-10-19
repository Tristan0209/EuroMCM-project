from dataclasses import asdict, replace
import numpy as np
from scipy.stats import qmc, triang
from .world import Scenario, rng_for

FORMAL_SEEDS = list(range(2030101800, 2030101812))
PILOT_SEEDS = [2028101800, 2028101801]
PROTOCOL_VERSION = 'Core-v2-2026-10-18'


def representative(n=7335):
    base = Scenario(n=n)
    return {
        'normal': base,
        'hard': replace(base, hard_fraction=.6),
        'compression': replace(base, compression=2.),
        'burst': replace(base, burst_factor=5.),
        'pre': replace(base, gain_bias=-.1, sigma_q=.1),
        'carbon': replace(base, sigma_carbon=.4, carbon_bias_fr=.4, carbon_bias_de=-.4),
        'tail': replace(base, sigma_service=.5),
        'fault': replace(base, outage='FR_L'),
    }


def joint(n=7335):
    s = representative(n)
    base = s['normal']
    return {'normal': base,
            'hard_pre': replace(s['hard'], gain_bias=-.1, sigma_q=.1),
            'load_burst': replace(s['compression'], burst_factor=5.),
            'carbon_fault': replace(s['carbon'], outage='FR_L'),
            'tail_fault': replace(s['tail'], outage='FR_L'),
            'hard_load': replace(s['hard'], compression=2.),
            'common_fault': replace(base, outage='all_L'),
            'all_joint': replace(base, hard_fraction=.6, compression=2., burst_factor=5., gain_bias=-.1,
                                 sigma_q=.1, sigma_carbon=.4, carbon_bias_fr=.4, carbon_bias_de=-.4,
                                 sigma_service=.5, outage='all_L')}


def conditional_mc(regime, seed):
    if regime not in [0, 1]:
        raise ValueError('Binary regime required')
    rng = rng_for(seed, 'conditional_parameters')
    adverse = regime == 1
    ell = rng.uniform(1.2, 2.) if adverse else rng.uniform(.8, 1.2)
    hard = rng.uniform(.4, .6) if adverse else rng.uniform(1796/7335, .4)
    bias = rng.uniform(-.1, 0) if adverse else rng.uniform(-.03, .03)
    sigmaq = rng.uniform(.05, .1) if adverse else rng.uniform(0, .05)
    sigmac = rng.uniform(.15, .4) if adverse else rng.uniform(0, .15)
    sigmat = rng.uniform(.3, .5) if adverse else rng.uniform(.1, .25)
    burst = (5. if adverse else 3.) if rng.random() < (.5 if adverse else .1) else 1.
    outage = 'none'
    if rng.random() < (.5 if adverse else .05):
        outage = 'all_L' if adverse and rng.random() < .25 else 'FR_L'
    start = rng.uniform(1800, 10800)
    duration = rng.uniform(900, 1800) if adverse else rng.uniform(300, 900)
    return Scenario(compression=ell, hard_fraction=hard, gain_bias=bias, sigma_q=sigmaq,
                    sigma_carbon=sigmac, carbon_bias_fr=sigmac, carbon_bias_de=-sigmac,
                    sigma_service=sigmat, burst_factor=burst, outage=outage,
                    outage_start=start, outage_duration=duration)


LHS_NAMES = ['shared_w', 'gpu_idle_w', 'gpu_busy_w', 'pue', 'service_scale_s', 'service_scale_l',
             'sigma_service', 'carbon_scale', 'carbon_contrast', 'network_delay_scale',
             'network_energy_scale', 'controller_w', 'router_seconds', 'sigma_carbon', 'sigma_q']


def lhs_points(n=16, seed=20311018):
    unit = qmc.LatinHypercube(d=15, seed=seed).random(n)
    def tri(u, a, mode, b):
        return float(triang.ppf(u, (mode-a)/(b-a), loc=a, scale=b-a))
    scenarios = []
    for u in unit:
        x = [tri(u[0], 750, 1500, 2500), 100+150*u[1], tri(u[2], 500, 760, 1000),
             tri(u[3], 1.1, 1.2, 1.6), tri(u[4], .7, 1, 1.5), tri(u[5], .7, 1, 1.5),
             .05+.45*u[6], .8+.4*u[7], 1.5*u[8], .5*6**u[9], .1*100**u[10],
             5*10**u[11], tri(u[12], .005, .01, .05), .4*u[13], .1*u[14]]
        s = Scenario(**dict(zip(LHS_NAMES, x)))
        s.validate()
        scenarios.append(s)
    return unit, scenarios


def core_plan():
    rows = []
    def add(module, name, scenario, seeds, policies=('Greedy', 'Spatial_MILP'), **metadata):
        for seed in seeds:
            for policy in policies:
                rows.append({'module': module, 'scenario_name': name, 'scenario': asdict(scenario),
                             'seed': seed, 'policy': policy, **metadata})
    for name, s in representative().items():
        add('C1', name, s, FORMAL_SEEDS[:3])
    for name, s in joint().items():
        add('C2', name, s, FORMAL_SEEDS)
    for regime in [0, 1]:
        for j in range(1000):
            seed = 2040101800 + regime*10000 + j
            add('C3', 'regime_' + str(regime), conditional_mc(regime, seed), [seed], regime=regime)
    for r in [0., .25, .5, 1., 2.]:
        for hosts in [1, 2]:
            add('C4', f'growth_{r}_Lhosts_{hosts}', Scenario(growth=r, l_hosts=hosts), FORMAL_SEEDS[:6])
    for scramble in range(2):
        _, points = lhs_points(seed=20311018+scramble)
        for j, s in enumerate(points):
            add('C6', f'lhs_{scramble}_{j}', s, FORMAL_SEEDS[:3], scramble=scramble, parameter_point=j)
    add('C8', 'all_l_normal', Scenario(), FORMAL_SEEDS[:6], policies=('AllL_Greedy',))
    assert len(rows) == 4558
    return rows


def pilot_config():
    s = representative(800)
    scenarios = [dict(name='normal', parameters=asdict(s['normal'])),
                 dict(name='pre_single', parameters=asdict(s['pre'])),
                 dict(name='load_single_fault', parameters=asdict(replace(s['fault'], compression=2.))),
                 dict(name='hard_pre', parameters=asdict(replace(s['hard'], gain_bias=-.1, sigma_q=.1))),
                 dict(name='carbon_reduced', parameters=asdict(replace(s['carbon'], l_hosts=2, outage='FR_L',
                      reduced_fraction=.5, outage_duration=10800.))),
                 dict(name='common_fault_tail', parameters=asdict(replace(s['tail'], outage='all_L'))),
                 dict(name='growth_hosts', parameters=asdict(Scenario(n=800, growth=.25, l_hosts=2)))]
    return {'version': 'bounded-pilot-v3', 'formal_enabled': False, 'seeds': PILOT_SEEDS,
            'policies': ['Greedy', 'Spatial_MILP'], 'scenarios': scenarios,
            'purpose': 'engineering validation only; no formal distribution tuning'}
