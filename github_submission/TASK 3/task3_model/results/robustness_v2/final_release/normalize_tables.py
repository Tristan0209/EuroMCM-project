from context import *
import pandas as pd
def main():
    headers=['strategy','seed','gamma','delay_scale','request_id','changed_fields','interpretation']
    headers += [c+s for c in ['region','status','dispatch_s','service_start_s','completion_s','intentional_delay_s','queue_wait_s','service_constraint_violation'] for s in ['_old','_new']]
    for phase in ['core','F','H']:
        p=out(phase)/'before_after_changed_requests.csv.gz'
        try:pd.read_csv(p)
        except pd.errors.EmptyDataError:pd.DataFrame(columns=headers).to_csv(p,index=False,compression='gzip')
if __name__=='__main__':main()
