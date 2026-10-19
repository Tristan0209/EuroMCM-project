from context import *
import argparse
import experiments
import pandas as pd
def main():
    p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,choices=[1,2],required=True);a=p.parse_args()
    original=experiments.read
    def overlay(path):
        value=original(path)
        if Path(path)==ROOT/'audit/source_and_input_hashes.json':
            value={**value,'fresh_repeat_marker':a.repeat}
        return value
    experiments.read=overlay
    rows=[experiments.run_case(policy=s,seed=SEEDS[6]) for s in ['Greedy','Spatial_MILP','MPC','E-Spatial','E-Full']]
    pd.DataFrame(rows).to_csv(ROOT/'audit'/('fresh_repeat_'+str(a.repeat)+'.csv'),index=False)
if __name__=='__main__':main()
