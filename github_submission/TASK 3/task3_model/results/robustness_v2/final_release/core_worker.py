from context import *
import argparse
import experiments
def main():
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True);a=p.parse_args()
    save(ROOT/'audit'/('ideal_worker_%d_%d.json'%(a.start,a.stop)),{
        'seed_indexes':list(range(a.start,a.stop)),'parallel_ideal_only':True,
        'note':'Only scale=0 cases. Wall measurements are descriptive, not used in ideal decisions. H remains isolated and sequential.'})
    for seed in SEEDS[a.start:a.stop]:
        for s in ['Greedy','Spatial_MILP','MPC','E-Spatial','E-Full']:experiments.run_case(s,seed)
if __name__=='__main__':main()
