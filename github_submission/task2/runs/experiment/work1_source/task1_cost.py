from __future__ import annotations
import argparse
import json
import heapq
from pathlib import Path
import numpy as np
import pandas as pd

TRACE_FILENAME = 'AzureLLMInferenceTrace_code_1week.csv'
PRICE_FILENAME = 'energy-charts_Electricity_production_and_spot_prices_in_all_available_countries_in_2026.csv'

def locate_input(filename):


    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    roots = list(dict.fromkeys([script_dir, cwd, *list(script_dir.parents)[:3], *list(cwd.parents)[:2]]))
    folders = ('', 'AI_datasets', 'AI_datasets/Azure', 'AI_datasets/Energy-Charts', 'AI_datasets/energy_charts',
               'Datasets', 'data', 'datasets', 'Azure')
    candidates = [root / folder / filename for root in roots for folder in folders]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    inspected = '\n'.join('  ' + str(root) for root in roots)
    raise FileNotFoundError(
        f'找不到原始文件：{filename}\n请把它放在task1_cost.py同一文件夹，'
        f'或者用 --trace/--prices 传入完整路径。\n已检查这些目录及其常见数据子目录：\n{inspected}'
    )

ZONES={'FR':'Day Ahead Auction (FR)','DE-LU':'Day Ahead Auction (DE-LU)','PL':'Day Ahead Auction (PL)'}


def load_prices(path, start, end):

    raw=pd.read_csv(path, skiprows=[0,2], usecols=['Date (GMT+1)',*ZONES.values()], low_memory=False)
    raw['time']=pd.to_datetime(raw['Date (GMT+1)'], utc=True, errors='coerce')
    if raw.time.isna().any(): raise ValueError('电价时间字段无法转换为UTC')
    raw=raw.sort_values('time').drop_duplicates('time',keep='last')
    if raw.time.isna().any() or raw.time.duplicated().any(): raise ValueError('电价时间无效/重复')
    for z,col in ZONES.items():
        raw[z]=pd.to_numeric(raw[col], errors='coerce')/1000
    raw=raw[(raw.time>=start-pd.Timedelta(minutes=15))&(raw.time<end+pd.Timedelta(minutes=15))].copy()
    if len(raw)<2: raise ValueError('价格覆盖日期不匹配。请将 --price-date 设置为CSV内的2026日期。')

    times=raw.time.to_numpy(dtype='datetime64[ns]').astype('int64')
    step=int(pd.Timedelta(minutes=15).value)
    if np.any(np.diff(times)!=step):
        raise ValueError('所选区间的电价时间戳不连续（非15分钟）。请检查源文件缺失/重复。')
    if times[0]>start.value or times[-1]+step<end.value:
        raise ValueError('所选电价未完整覆盖仿真运行时间')
    for z in ZONES:
        if not np.isfinite(raw[z]).all(): raise ValueError(f'电价{z}存在缺失/非法值，不能插值编造')
    return times, {z:raw[z].to_numpy(float) for z in ZONES}, len(raw)


def load_trace(path, n):

    df=pd.read_csv(path,usecols=['TIMESTAMP','ContextTokens','GeneratedTokens'],nrows=n,low_memory=False)
    a=pd.to_datetime(df.TIMESTAMP,utc=True,errors='coerce')
    l=pd.to_numeric(df.ContextTokens,errors='coerce')
    o=pd.to_numeric(df.GeneratedTokens,errors='coerce')
    valid=a.notna() & l.notna() & o.notna() & (l>=0) & (o>=1)
    df=pd.DataFrame({'arrival':a[valid],'L':l[valid],'O':o[valid]}).sort_values('arrival',kind='stable')
    if len(df)<2: raise ValueError('有效Azure请求不足2条')
    if (df.arrival.diff().dropna()<pd.Timedelta(0)).any(): raise ValueError('请求顺序异常')
    return df.reset_index(drop=True), {'raw_rows_read':len(valid),'eligible_rows':len(df),'excluded_rows':int((~valid).sum())}


def sample_service(df, seed, sigma):
    x=df.L.to_numpy(float)/1000
    y=(df.O.to_numpy(float)-1)/1000

    first=.05+.2*x+.01*x*x
    decode=8*y+.1*x*y+.02*y*y
    rng=np.random.default_rng(seed)
    shock=np.exp(sigma*rng.standard_normal(len(df))-sigma*sigma/2)
    return first*shock, (first+decode)*shock, first+decode


def schedule(arrivals, service, replicas):

    free=[(0.0,i) for i in range(replicas)]
    heapq.heapify(free)
    events=[]
    waiting=[]
    finishes=[]
    for a,s in zip(arrivals,service):
        avail,j=heapq.heappop(free)
        begin=max(float(a),avail)
        done=begin+float(s)
        heapq.heappush(free,(done,j))
        events.append((begin,+1));events.append((done,-1))
        waiting.append(begin-a);finishes.append(done)
    return events, np.asarray(waiting), np.asarray(finishes)


def server_power_w(busy, *, total_gpus, shared_w, idle_w, busy_w, power_cap_w):





    if not (0 <= busy <= total_gpus):
        raise ValueError('活跃 GPU 数越界')
    watts = shared_w + total_gpus * idle_w + busy * (busy_w - idle_w)
    if watts > power_cap_w + 1e-8:
        raise ValueError(f'整机假设功率 {watts:.1f} W 超过官方 {power_cap_w:.1f} W 参考上限')
    return watts


def integrate(events, horizon, replicas, anchor, price_ns, price_vectors,
              pue, idle_w, busy_w, *, server_gpu_count, shared_w, power_cap_w):

    ev=sorted(events)
    bounds={0.0,float(horizon)}
    for t,_ in ev:
        if 0 <= t <= horizon: bounds.add(float(t))
    absolute_start=anchor.value
    for ns in price_ns:
        sec=(int(ns)-absolute_start)/1e9
        if 0 < sec < horizon: bounds.add(sec)
    b=sorted(bounds)
    busy=0;pos=0
    gpu_kwh=0.0; it_kwh=0.0; facility_kwh=0.0
    fees={z:0.0 for z in ZONES}
    for a,e in zip(b[:-1],b[1:]):
        while pos<len(ev) and ev[pos][0]<=a+1e-10:
            busy+=ev[pos][1];pos+=1
        if busy<0 or busy>replicas: raise ValueError('FCFS事件功率区间异常')
        if e<=a: continue
        dt=e-a
        watts=server_power_w(busy,total_gpus=server_gpu_count,
                shared_w=shared_w,idle_w=idle_w,busy_w=busy_w,power_cap_w=power_cap_w)
        gpu_watts=server_gpu_count*idle_w+(busy_w-idle_w)*busy
        gpu_kwh+=gpu_watts*dt/3_600_000
        it_kwh+=watts*dt/3_600_000
        e_kwh=watts*dt*pue/3_600_000
        facility_kwh+=e_kwh
        midpoint=absolute_start+int((a+e)/2*1e9)
        idx=int(np.searchsorted(price_ns,midpoint,side='right')-1)
        if idx<0 or idx>=len(price_ns): raise ValueError('电价覆盖不足')
        for z in ZONES: fees[z]+=e_kwh*price_vectors[z][idx]
    return facility_kwh,fees,it_kwh,gpu_kwh



def run_validation(results, inputs, args, dest, price_vectors):











    directory=dest.parent
    checks=[]
    scenario={(float(x['arrival_multiplier']),int(x['replicas'])):x for x in inputs}
    for _,r in results.iterrows():
        x=scenario[(float(r['arrival_multiplier']),int(r['replicas']))]
        tag={'region':r['region'],'arrival_multiplier':r['arrival_multiplier'],'replicas':int(r['replicas'])}
        energy_error=float(r['facility_energy_kwh'])-x['energy_independent']
        cost_error=float(r['cost_total_eur'])-(r['cost_electricity_eur']+r['cost_hardware_eur']+r['cost_maintenance_eur'])
        expected_hw=(args.dgx_price_usd*args.usd_to_eur)/(args.lifetime_years*365*24)*args.dgx_share_per_replica*int(r['replicas'])*r['simulation_duration_s']/3600
        hw_error=float(r['cost_hardware_eur'])-expected_hw
        relative_energy=abs(energy_error)/max(abs(x['energy_independent']),1e-12)

        mean_spot=float(r['cost_electricity_eur'])/float(r['facility_energy_kwh'])
        price_low=float(np.min(price_vectors[r['region']]))
        price_high=float(np.max(price_vectors[r['region']]))
        price_bounds_ok=price_low-1e-10<=mean_spot<=price_high+1e-10
        checks.append({**tag,'mean_paid_spot_eur_per_kwh':mean_spot,
            'price_min_eur_per_kwh':price_low,'price_max_eur_per_kwh':price_high,
            'pass_price_bounds':bool(price_bounds_ok),
            'energy_absolute_error_kwh':abs(energy_error),
            'energy_relative_error':relative_energy,'cost_identity_absolute_error_eur':abs(cost_error),
            'depreciation_absolute_error_eur':abs(hw_error),
            'positive_energy':bool(r['facility_energy_kwh']>0),
            'pass_energy':bool(relative_energy<1e-8),
            'pass_cost_identity':bool(abs(cost_error)<1e-9),
            'pass_depreciation':bool(abs(hw_error)<1e-9)})
    chk=pd.DataFrame(checks)
    chk.to_csv(directory/'validation_checks.csv',index=False,encoding='utf-8-sig')



    correlation=[]
    for region,g in results.groupby('region',sort=False):
        for left,right in [('cost_total_eur','p95_response_s'),
                           ('cost_total_eur','mean_response_s'),
                           ('cost_electricity_eur','facility_energy_kwh'),
                           ('cost_hardware_eur','simulation_duration_s')]:
            for method in ['pearson','spearman']:
                coef=float(g[left].corr(g[right],method=method)) if g[left].nunique()>1 and g[right].nunique()>1 else float('nan')
                correlation.append({'region':region,'x':left,'y':right,'method':method,
                                    'n_configurations':len(g),'coefficient':coef,
                                    'meaning':'scenario association, NOT prediction accuracy'})
    cor=pd.DataFrame(correlation)
    cor.to_csv(directory/'validation_correlations.csv',index=False,encoding='utf-8-sig')


    sensitivities=[]
    for _,r in results.iterrows():
        for lifetime in [4,5,6]:
            for price_scale in [.5,1.0,1.5]:
                for capex_scale in [.75,1.0,1.25]:
                    hw=r['cost_hardware_eur']*capex_scale*args.lifetime_years/lifetime
                    electric=r['cost_electricity_eur']*price_scale
                    sensitivities.append({'region':r['region'],'arrival_multiplier':r['arrival_multiplier'],
                                          'replicas':r['replicas'],'lifetime_years':lifetime,
                                          'capex_multiplier':capex_scale,'spot_price_multiplier':price_scale,
                                          'scenario_total_eur':hw+electric+r['cost_maintenance_eur']})
    pd.DataFrame(sensitivities).to_csv(directory/'validation_sensitivity.csv',index=False,encoding='utf-8-sig')


    report={
      'n_scenarios':int(len(results)), 'n_independent_arrival_replica_configurations':int(len(scenario)),
      'checks_pass':bool(chk[['pass_energy','pass_cost_identity','pass_depreciation','positive_energy','pass_price_bounds']].all().all()),
      'max_energy_relative_error':float(chk['energy_relative_error'].max()),
      'max_cost_identity_absolute_error_eur':float(chk['cost_identity_absolute_error_eur'].max()),
      'hardware_cost_share_min':float((results['cost_hardware_eur']/results['cost_total_eur']).min()),
      'hardware_cost_share_max':float((results['cost_hardware_eur']/results['cost_total_eur']).max()),
      'correlation_warning':'Same nine configurations are repriced for three regions; pooled 27-row p-values are inappropriate. Formulas induce some correlations.',
      'external_validation_status':'NOT AVAILABLE: No matched real DGX power/energy bills or same-request RouterBench API costs.',
      'time_warning':'Azure 2024 arrivals scaled and mapped to 2026 electricity market: counterfactual, not historical co-observation.',
      'spot_price_warning':'Day-ahead wholesale energy price is not final datacenter retail tariff.',
      'equipment_warning':'Full physical DGX electricity is counted, but hardware depreciation is still allocated per GPU replica (unchanged); these are mismatched cost boundaries until CAPEX accounting is aligned.',
      'sensitivity_warning':'CAPEX/depreciation/price are coefficient sensitivities; separate power_sensitivity.csv re-integrates time-varying electricity for shared/idle assumptions with unchanged queueing.'}
    (directory/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figs=directory/'validation_figures';figs.mkdir(exist_ok=True)

        for region,g in results.groupby('region',sort=False):
            fig,ax=plt.subplots(figsize=(7,5))
            for replicas,z in g.groupby('replicas'):
                ax.scatter(z['p95_response_s'],z['cost_per_1000_requests_eur'],label=f'{replicas} replicas')
            ax.set(xlabel='P95 response (s)',ylabel='Cost (EUR / 1000 requests)',title=f'{region}: cost vs latency')
            ax.legend();fig.tight_layout();fig.savefig(figs/f'{region.replace("-","_")}_cost_latency.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        base=results[results['arrival_multiplier']==1.0]
        for region,g in base.groupby('region',sort=False):
            ax.plot(g['replicas'],g['cost_total_eur'],marker='o',label=region)
        ax.set(xlabel='Replicas',ylabel='Total cost (EUR)',title='Baseline load: deployment cost')
        ax.legend();fig.tight_layout();fig.savefig(figs/'baseline_cost_by_region.png',dpi=180);plt.close(fig)
    except ImportError:
        report['charts_note']='matplotlib missing; validation CSV/JSON still produced. pip install matplotlib'
        (directory/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('验证完成：', 'PASS' if report['checks_pass'] else 'FAIL',
          '| 最大能量相对误差:',f"{report['max_energy_relative_error']:.2e}")
    print('验证文件目录:',directory.resolve())
    if not report['checks_pass']:
        raise AssertionError('Cost核算一致性未通过，请检查 validation_checks.csv')


def main():
    parser=argparse.ArgumentParser(description='Task1 Cost：原始Azure + 原始Energy-Charts，一键清洗与模拟')
    parser.add_argument('--trace',default=None,help='Azure原始CSV；不填则自动查找文件名')
    parser.add_argument('--prices',default=None,help='Energy-Charts原始CSV；不填则自动查找文件名')
    parser.add_argument('--usd-to-eur',type=float,default=0.90,help='固定情景假设1 USD = 0.90 EUR（非实时汇率）')
    parser.add_argument('--dgx-price-usd',type=float,default=515000,help='整台DGX B200参考价，未验证是否官方')
    parser.add_argument('--dgx-share-per-replica',type=float,default=1/8,help='抽象副本占用整机CAPEX份额，默认为1/8的情景假设')
    parser.add_argument('--lifetime-years',type=float,default=5)
    parser.add_argument('--maintenance-per-replica-hour',type=float,default=0)
    parser.add_argument('--rows',type=int,default=5000,help='Azure每个文件读取的前N行')
    parser.add_argument('--price-date',default='2026-05-12',help='映射到的2026电价起点(UTC日期)')
    parser.add_argument('--seed',type=int,default=20261016)
    parser.add_argument('--sigma',type=float,default=.15)
    parser.add_argument('--replicas',nargs='+',type=int,default=[1,2,4])
    parser.add_argument('--loads',nargs='+',type=float,default=[.5,1,2])
    parser.add_argument('--rho-ref',type=float,default=.6)
    parser.add_argument('--pue',type=float,default=1.2)
    parser.add_argument('--idle-w',type=float,default=175,help='单块B200空闲功率W，情景假设，非实测')
    parser.add_argument('--busy-w',type=float,default=760,help='单块B200忙碌均值W；默认来自上传Qwen3-14B样本配置，仅作匹配情景')
    parser.add_argument('--server-gpu-count',type=int,default=8,help='DGX B200整机GPU数量')
    parser.add_argument('--shared-w',type=float,default=1500,help='整机共享底座功率W；未经实测的中值情景，不是B200已验证参数')
    parser.add_argument('--shared-sensitivity',nargs='+',type=float,default=[750,1500,2500],help='整机非GPU共享功率的低/中/高情景W')
    parser.add_argument('--gpu-idle-sensitivity',nargs='+',type=float,default=[100,175,250],help='单GPU空闲功率的低/中/高情景W')
    parser.add_argument('--server-max-w',type=float,default=14300,help='NVIDIA官网DGX B200整机最大功率W，非平均值')
    parser.add_argument('--validate', action=argparse.BooleanOptionalAction, default=True, help='自动运行数值校验、相关性及敏感性实验')
    parser.add_argument('--out',default=str(Path(__file__).resolve().parent / 'outputs' / 'cost_summary.csv'))
    args=parser.parse_args()
    args.trace = args.trace or locate_input(TRACE_FILENAME)
    args.prices = args.prices or locate_input(PRICE_FILENAME)
    print(f'Azure Trace: {args.trace}')
    print(f'Energy-Charts: {args.prices}')
    print(f'固定汇率: 1 USD = {args.usd_to_eur} EUR (建模假设)')
    print(f'整机功率模型: DGX {args.server_gpu_count}GPU, shared={args.shared_w}W(假设), GPU idle={args.idle_w}W(假设), GPU active={args.busy_w:.2f}W(特定基准配置); ceiling={args.server_max_w}W')
    if args.usd_to_eur<=0 or args.dgx_price_usd<=0 or args.lifetime_years<=0 or not 0<args.dgx_share_per_replica<=1:raise ValueError('价格、折旧寿命、汇率、资源份额必须为正')
    if args.idle_w<0 or args.busy_w<args.idle_w or args.pue<1:raise ValueError('要求 0 <= Idle 功率 <= Busy 功率，且 PUE >= 1')
    if min(args.replicas)<=0 or min(args.loads)<=0:raise ValueError('副本数、到达倍数必须为正')
    if max(args.replicas)>args.server_gpu_count or args.server_gpu_count!=8 or args.shared_w<0 or min(args.shared_sensitivity)<0 or min(args.gpu_idle_sensitivity)<0 or max(args.gpu_idle_sensitivity)>args.busy_w: raise ValueError('本版对应一台8卡DGX：1 <= 副本数 <= 8；共享功率非负')
    server_power_w(args.server_gpu_count,total_gpus=args.server_gpu_count,shared_w=args.shared_w,idle_w=args.idle_w,busy_w=args.busy_w,power_cap_w=args.server_max_w)
    df,audit=load_trace(args.trace,args.rows)
    first,service,det=sample_service(df,args.seed,args.sigma)
    orig=(df.arrival-df.arrival.iloc[0]).dt.total_seconds().to_numpy()
    span=orig[-1]
    if span<=0:raise ValueError('Azure时间跨度为0，无法缩放')

    scale=((len(df)-1)*float(det.mean())/args.rho_ref)/span
    anchor=pd.Timestamp(args.price_date,tz='UTC')

    cached=[]
    for load in args.loads:
        arrivals=orig*scale/load
        for c in args.replicas:
            ev,waiting,finish=schedule(arrivals,service,c)
            cached.append((load,c,ev,waiting,float(finish.max())))
    max_horizon=max(item[4] for item in cached)
    prices_ns,prices,nprice=load_prices(args.prices,anchor,anchor+pd.Timedelta(seconds=max_horizon))
    rate_per_replica=(args.dgx_price_usd*args.usd_to_eur*args.dgx_share_per_replica)/(args.lifetime_years*8760)
    rows=[]
    validation_inputs=[]
    for load,c,ev,waiting,horizon in cached:
        energy,fees,it_energy,gpu_energy=integrate(ev,horizon,c,anchor,prices_ns,prices,args.pue,args.idle_w,args.busy_w,server_gpu_count=args.server_gpu_count,shared_w=args.shared_w,power_cap_w=args.server_max_w)
        hw=c*(horizon/3600)*rate_per_replica
        maintenance=c*(horizon/3600)*args.maintenance_per_replica_hour


        expected_kwh=((args.shared_w+args.server_gpu_count*args.idle_w)*horizon+(args.busy_w-args.idle_w)*float(service.sum()))*args.pue/3_600_000
        validation_inputs.append({'arrival_multiplier':load,'replicas':c,'energy_integrated':energy,
            'energy_independent':expected_kwh,'duration_s':horizon,'hw_eur':hw,
            'maintenance_eur':maintenance,'n_requests':len(df)})
        for region,electric in fees.items():
            total=hw+maintenance+electric
            rows.append({'region':region,'arrival_multiplier':load,'replicas':c,'n_requests':len(df),
                'simulation_duration_s':horizon,'facility_energy_kwh':energy,'server_it_energy_kwh':it_energy,'gpu_energy_kwh':gpu_energy,'shared_equipment_energy_kwh':args.shared_w*horizon/3_600_000,'p95_response_s':float(np.quantile(waiting+service,.95)),
                'mean_response_s':float(np.mean(waiting+service)),
                'cost_electricity_eur':electric,'cost_hardware_eur':hw,'cost_maintenance_eur':maintenance,
                'cost_total_eur':total,'cost_per_1000_requests_eur':total*1000/len(df)})


    power_sensitivity_rows=[]
    for load,c,ev,waiting,horizon in cached:
        hw=c*(horizon/3600)*rate_per_replica
        maintenance=c*(horizon/3600)*args.maintenance_per_replica_hour
        for shared in args.shared_sensitivity:
            for gpu_idle in args.gpu_idle_sensitivity:
                try:
                    e_fac,fees_s,it_e,gpu_e=integrate(
                        ev,horizon,c,anchor,prices_ns,prices,args.pue,gpu_idle,args.busy_w,
                        server_gpu_count=args.server_gpu_count,shared_w=shared,power_cap_w=args.server_max_w)
                except ValueError as exc:

                    for region in ZONES:
                        power_sensitivity_rows.append(dict(region=region,arrival_multiplier=load,replicas=c,
                            shared_w=shared,gpu_idle_w=gpu_idle,gpu_busy_w=args.busy_w,
                            feasible=False,reason=str(exc)))
                    continue
                for region in ZONES:
                    power_sensitivity_rows.append(dict(region=region,arrival_multiplier=load,replicas=c,
                        shared_w=shared,gpu_idle_w=gpu_idle,gpu_busy_w=args.busy_w,
                        feasible=True,reason='',facility_energy_kwh=e_fac,
                        cost_electricity_eur=fees_s[region],cost_hardware_eur=hw,
                        cost_maintenance_eur=maintenance,cost_total_eur=fees_s[region]+hw+maintenance))
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest,index=False,encoding='utf-8-sig')
    pd.DataFrame(power_sensitivity_rows).to_csv(dest.parent/'power_sensitivity.csv',index=False,encoding='utf-8-sig')
    meta={**vars(args),'trace_audit':audit,'valid_electricity_price_rows_used':nprice,
      'disclaimers':['2024 Azure workload time pattern is shifted/rescaled to 2026 prices: counterfactual, not observed',
      'DGX B200 515000 USD is unverified market reference, 1/8 per abstract replica is allocation scenario',
      f'Power model: full {args.server_gpu_count}-GPU server, P_shared={args.shared_w}W assumed; GPU idle={args.idle_w}W assumed; active={args.busy_w}W benchmark configuration only; capped at {args.server_max_w}W; NOT whole-server measured power',
      'Full DGX electricity charged even when only 1/2/4 GPUs handle requests; hardware depreciation formula remains per-replica 1/8 DGX (unequal accounting boundary; avoid interpreting as full physical DGX TCO)',
      'power_sensitivity.csv tests shared 750/1500/2500 W and idle 100/175/250 W as scenarios, NOT measured DGX power',
      'NVIDIA maximum 14300W is a system upper limit only, not a measured mean. Barroso & Hölzle 2007 supports non-proportional idle power concept, not 1500W on DGX',
      'source Energy-Charts license statement must be reviewed before redistribution; negatives are preserved',
      'output tokens known only for offline assessment, not at routing time; hourly purchased price is wholesale, not retail bill']}
    dest.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.validate:
        run_validation(pd.DataFrame(rows), validation_inputs, args, dest, prices)
    print('处理完成，结果:',dest.resolve())
    print(pd.DataFrame(rows)[['region','arrival_multiplier','replicas','cost_electricity_eur','cost_hardware_eur','cost_total_eur']].to_string(index=False))

if __name__=='__main__':main()
