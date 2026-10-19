import re
import pandas as pd
import matplotlib.pyplot as plt
from .util import ROOT,DATA,read


def table(frame):
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join(
        '| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def run():
    config=read(ROOT/'configs/upstream.json');archive=ROOT/config['legacy_archive']
    old=pd.read_csv(archive/'results/experiment_main.csv').set_index(['strategy','rep'])
    new=pd.read_csv(ROOT/'results/experiment_main.csv').set_index(['strategy','rep'])
    contract=read(DATA/'resource_accounting_contract.json');p=contract['profile']
    base=p['shared_w']+p['gpu_count_per_server']*p['gpu_idle_w']
    delta=p['gpu_busy_w']-p['gpu_idle_w']
    rows=[]
    for key,a in old.iterrows():
        b=new.loc[key]
        rows.append({'strategy':key[0],'rep':int(key[1]),'old_co2e_g':a.co2e_g,'old_energy_kwh':a.energy_kwh,
            'old_plan_new_power_co2e_g':a.idle_co2e_g*base/175+a.active_increment_co2e_g*delta/75+a.network_co2e_g+a.controller_co2e_g,
            'old_plan_new_power_energy_kwh':a.idle_kwh*base/175+a.active_increment_kwh*delta/75+a.network_kwh+a.controller_kwh,
            'rerun_DGX_co2e_g':b.co2e_g,'rerun_DGX_energy_kwh':b.energy_kwh,
            'old_p95_s':a.p95_response_s,'rerun_DGX_p95_s':b.p95_response_s,
            'old_violation_rate':a.service_violation_rate,'rerun_DGX_violation_rate':b.service_violation_rate,
            'old_run_id':a.run_id,'rerun_run_id':b.run_id})
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'results/power_migration_comparison.csv',index=False)
    order=['Fixed','Greedy','Oracle','MPC'];g=frame.groupby('strategy').mean(numeric_only=True).reindex(order)
    show=pd.DataFrame({'策略':order,'旧口径CO2e/g':g.old_co2e_g.round(6),
        '旧计划新功率计价/g':g.old_plan_new_power_co2e_g.round(6),
        '新版重新调度/g':g.rerun_DGX_co2e_g.round(6),'新版电量/kWh':g.rerun_DGX_energy_kwh.round(6),
        '新版P95/s':g.rerun_DGX_p95_s.round(4),'新版违反/%':(g.rerun_DGX_violation_rate*100).round(4)})
    pairs=[]
    for baseline in ['Fixed','Greedy']:
        for column,label in [('old_co2e_g','旧功率实验'),('old_plan_new_power_co2e_g','旧计划新功率计价'),('rerun_DGX_co2e_g','新功率重新调度')]:
            diff=g.loc[baseline,column]-g.loc['MPC',column]
            pairs.append({'对照':baseline,'结果类型':label,'净减排/g':round(diff,6),
                '总量减排/%':round(100*diff/g.loc[baseline,column],6)})
    report=f'''# 第三问DGX功率更新说明

## 更新边界

上游采用当前task2/runs/experiment及Work 1配置，只读验证。旧第三问代码、配置、逐请求实验、图表、报告和PDF已归档于{config['legacy_archive']}，不与本轮实验混合。旧数据准备包保留为冻结基准，新版源适配与派生输入全部在task3_model内。

六台常开8-GPU主机：DE/FR/ES各S/L一台；每池仍只准入一个GPU服务槽位。安装48张GPU不意味着48个可调度副本。每台共享1500W、单GPU空闲175W、忙碌760W、PUE1.2；空闲整机2900W、单槽位运行时3485W，12小时固定空闲电量250.56kWh。不可用仅阻止新准入，主机仍开机。物理主机底座只计一次，增加准入槽位不复制主机。服务器配置中的busy/idle是整机单槽位状态的兼容别名，差值585W才是每个运行请求的IT增量。

## 冻结与验证

7335个请求ID、时间顺序/间隔、截止/等待类别、priority、来源地区、S/L、质量预测、服务时间和公共随机扰动均保持。唯一变化的原请求字段为estimated_energy_kwh，现为585×PUE×估计服务秒/3.6e6。全部14670个S/L质量预测通过当前已保存Pre模型重放，未重新训练或修改阈值。Code/Math/未知来源回退L保留。

21项原合成接口测试仍执行；原全数据验证器的旧路径/旧功率字段明确标为被当前源验证替代。新源检查重放路由、训练长度、服务代理和能耗。新输入指纹包含上游配置/源码/权重/路由、物理主机表与资源合同；上游更改后必须先audit，不能继续消费旧派生数据或缓存。

## 三类数字必须区分

下表均为完整7335请求、三个公共扰动重复的均值。“旧计划新功率计价”只对归档账本重计固定底座与增量；“新版重新调度”来自实际重跑MILP-MPC、其他策略及DES。当前正式结果采用最后一类。

{table(show.reset_index(drop=True))}

{table(pd.DataFrame(pairs))}

即使使用相同请求和服务扰动，新的增量碳/网络权衡与碳量化会改变MILP地区/开始时刻，最终完成和违反率也可能变化。不能用解析计价表替代新版策略实验，也不能把固定计划不变的延迟当成重新优化后的延迟。

## 数学与实验解释

对每台主机，P_IT(t)=P_shared+8P_GPU,idle+(P_GPU,busy-P_GPU,idle)b(t)。固定总能量为所有主机空闲底座加各请求忙碌增量，再加网络及单独控制。固定主机与12小时窗口使共享底座对候选决策是常数，但它必须进入设施总碳分母。碳按实际地区/小时积分，L服务时长不再乘3。现有MILP唯一分配、非抢占占用、最大等待、完成截止和因果预测约束保留。

边界、预测误差、需求及故障结果均重新运行，见RESULTS_zh.md与boundary_classification.csv；不得沿用旧版0.70268%或旧失效阈值。安装GPU、服务槽位、主机数量、观察窗口在策略间完全一致。正碳收益同时需满足共同完成输入与服务约束，非零违反不能判作严格部署合格。

## 操作与限制

在task3_model目录运行python -B -X utf8 run_all.py --phase all；仅刷新已完成新版图文报告用--phase report。按阶段的audit/test/pilot/scale/experiments/full-stress命令不变。正式上游位置在configs/upstream.json，模型参数在configs/core.json。更新PDF位于output/pdf_dgx；旧PDF和旧对话解释属于旧功率版本。

1500W共享/175W空闲为假设，760W为特定公开基准配置代理；不是GPT-4/Mixtral实测。真实碳数据不变，GWP具体快照版本和as-of历史仍未验证，不能称真实部署减排。旧全数据验证器没有被伪装为通过；新版具体测试数量、求解状态和完成组合数以VALIDATION_REPORT.md为准。
'''
    (ROOT/'reports/DGX功率更新说明.md').write_text(report,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=list(range(4))
    for offset,column,label,color in [(-.25,'old_co2e_g','Legacy power','#94a3b8'),
        (0,'old_plan_new_power_co2e_g','Same old plans repriced','#d48b18'),(.25,'rerun_DGX_co2e_g','DGX reoptimized DES','#326bc4')]:
        axes[0].bar([v+offset for v in x],g[column],width=.24,label=label,color=color)
    axes[0].set(xticks=x,xticklabels=order,yscale='log',ylabel='Total CO2e (g, log scale)')
    axes[0].legend(fontsize=8)
    for i,baseline in enumerate(['Fixed','Greedy']):
        axes[1].bar([i-.25,i,i+.25],[g.loc[baseline,c]-g.loc['MPC',c] for c in
            ['old_co2e_g','old_plan_new_power_co2e_g','rerun_DGX_co2e_g']],width=.24,
            color=['#94a3b8','#d48b18','#326bc4'])
    axes[1].set(xticks=[0,1],xticklabels=['vs Fixed','vs Greedy'],ylabel='Baseline minus MPC CO2e (g)')
    fig.suptitle('Power migration: repricing is distinct from reoptimization')
    fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(ROOT/'figures'/('11_power_migration_comparison.'+suffix),dpi=260,bbox_inches='tight')
    plt.close(fig)


    guide=(archive/'reports/第三问完整求解与运行指南.md').read_text(encoding='utf-8')
    guide=guide.replace('每副本空闲175W、忙碌250W、PUE1.2','每池一台常开8-GPU主机，整机空闲2900W、单槽位忙碌3485W、PUE1.2；只准入1个服务槽位')
    guide=guide.replace('327项上游保护文件SHA256保持不变','当前版本上游文件SHA256保持不变；旧327项快照保存在归档')
    validation=read(ROOT/'results/audit/final_validation.json')
    test_count=validation['model_tests']
    guide=guide.replace('系统18项','系统'+str(test_count)+'项')
    solver=validation['solver'];statuses=solver['status_counts']
    solver_paragraph=(f"汇总{solver['solver_decisions']}次优化决策包含单请求枚举，不全部是HiGHS调用。"
        f"status=0共{statuses.get('0',0)}、不可行/无候选{statuses.get('2',0)}、超时{statuses.get('1',0)}；"
        f"均有日志与回退。最大单批{solver['max_variables']}变量、{solver['max_constraints']}约束。"
        "没有建立7335请求逐秒全时域巨型MILP，也没有声称其全局最优。")
    guide=re.sub(r'汇总130229次优化决策.*?全局最优。',lambda m:solver_paragraph,guide,flags=re.S)
    guide=guide.replace('10组PNG/SVG','11组PNG/SVG').replace('十组图各回答什么','十一组图各回答什么')
    guide=guide.replace('\n\nPNG适合快速查看，SVG适合论文排版缩放。',
        '\n| 11_power_migration_comparison | 区分旧结果、旧计划新功率计价、新功率重新调度 |\n\nPNG适合快速查看，SVG适合论文排版缩放。')
    guide=guide.replace('\n\n本文解释','\n\n当前为DGX整机资源版本；旧175/250W结果已归档。当前输入在results/data，上游接口在configs/upstream.json，新版PDF在output/pdf_dgx；不要运行旧数据准备生成器来覆盖基准。导出PDF用python -B -X utf8 pdf_export/export_pdfs.py。\n\n本文解释',1)
    energy='''## 五、统一能耗与碳账本

本版采用六台常开8-GPU主机、六个准入槽位，共同观察窗口仍为12小时。单主机IT功率=1500+8×175+(760−175)×忙碌槽位数。固定空闲电量250.56kWh，活动IT增量585W/忙碌槽位，再乘PUE1.2。共享功率及所有闲置GPU只计一次；增加准入槽位不增加主机底座。不可用不自动断电。

estimated_energy_kwh仅为服务秒×585×1.2/3.6e6。server_config的p_idle_w/p_busy_w为整机空闲2900W及单忙碌槽位3485W，差值585W。完整设施电量=固定主机底座+忙碌增量+网络+单独控制。碳依实际地区/小时分段积分，不把跨小时服务都归到开始时刻。L服务时长已经缩放，不再乘3。

路由、控制与网络假设保持原口径；控制10W是单独增量计算工作代理，主优化0.01秒/次是假设，真实wall另记，并有实测wall进入DES的敏感性。所有碳结果为条件仿真，不是实际部署实测。

'''
    guide=re.sub(r'## 五、统一能耗与碳账本.*?(?=## 六、)',lambda m:energy,guide,flags=re.S)
    results=(ROOT/'RESULTS_zh.md').read_text(encoding='utf-8')
    guide=re.sub(r'## 七、实际结果怎么解释.*?(?=## 八、)',lambda m:'## 七、实际结果怎么解释（本次重新求解）\n\n'+results+'\n\n',guide,flags=re.S)

    guide=guide.replace('设施忙碌增量功率90W，增量碳为 `20×90/3.6e6×30=0.015g`','设施忙碌增量功率702W，增量碳为 `20×702/3.6e6×30=0.117g`')
    start=guide.index('## 十、论文应当如何落结论')
    guide=guide[:start]+'''## 十、论文应当如何落结论

以第七章本次实际运行的碳差、延迟、完成与违反率回答RQ，不再引用旧175/250W场景的0.70%。区分固定计划重计价与新功率重新优化；比较方法和完整数据见DGX功率更新说明。六台常开主机使空闲底座占比很大，调度的活动增量收益不能替代设施总收益。若任一服务约束不满足，不能仅凭碳差为正判定严格部署值得。

真实碳曲线与仿真请求/功率/服务必须区分；没有全问题全局最优、真实商业模型硬件迁移或实际部署减排证明。确切GWP/as-of版本、匹配硬件测量及独立质量留出仍待补充。
'''
    (ROOT/'reports/第三问完整求解与运行指南.md').write_text(guide,encoding='utf-8')


if __name__=='__main__':run()
