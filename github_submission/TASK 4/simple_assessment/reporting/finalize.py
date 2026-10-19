import ast
import json
from pathlib import Path
import re
import shutil
import sys
import time
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(HERE),str(ROOT)]
import numpy as np
import pandas as pd
from task4_core.sources import read,sha
from task4_core.serialization import save
from task4_core.registry import reusable
from experiment import SEEDS


def main():
    checked = sorted((HERE/'audit').glob('independent_validation_*.json'))[-1]
    checks = read(checked)
    assert checks['status']=='PASS' and checks['executions']==24
    protection_file = sorted((HERE/'audit').glob('protection_and_contracts_*.json'))[-1]
    protection = read(protection_file)
    assert protection['status']=='PASS'
    log = (HERE/'audit/final_release_verify.log').read_text(encoding='utf-16' if (HERE/'audit/final_release_verify.log').read_bytes()[:2] in [b'\xff\xfe',b'\xfe\xff'] else 'utf-8-sig')
    assert 'PASS read-only verification' in log
    results = pd.read_csv(HERE/'Task4_Simple_Results.csv')
    assert len(results)==24 and not results[['scenario','seed','policy']].duplicated().any()
    assert np.array_equal(results.completed+results.failed+results.censored,results.requests)
    assert np.isfinite(results[['mean_historical_quality','answer_pass_rate','service_violation_rate','co2e_g','energy_kwh','p95_response_s']]).all().all()
    assert (results.censored==results.unfinished_at_H).all()
    paired = pd.read_csv(HERE/'Task4_Simple_Paired_Differences.csv')
    assert len(paired)==12
    for row in results.itertuples():
        record = read(row.source_record)
        assert reusable(record,record['config'])
        raw = read(Path(row.source_record).parent/'metrics.json')
        assert row.co2e_g==raw['co2e_g'] or np.isclose(row.co2e_g,raw['co2e_g'],rtol=0,atol=1e-9)
        assert np.isclose(row.energy_kwh,raw['energy_kwh'],rtol=0,atol=1e-9)
        assert abs(row.co2e_g-sum(raw[x+'_co2e_g'] for x in ['idle','active_increment','network','controller']))<1e-8
    bench = pd.read_csv(HERE/'audit/runtime_measurements.csv')
    size = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    ledger = [json.loads(line) for line in (HERE/'runs/ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    failures = [r for r in ledger if 'FAILED' in r['event']]
    assert not failures
    assert sum(r['event']=='COMPLETE_VALIDATED' for r in ledger)==24

    text = f'''# TASK 4 简化实验验证报告

日期：2026-10-18。**PASS：四情景、3seed、2策略共24次完整执行，另有8次80/120实例的小规模先行检查。**没有运行旧C1–C8、CVaR、LHS或新增优化模型。以下验证针对本次实际工件，旧42合同/Pilot作为保留的既有证据，没有把它们计成本轮新样本。

## 1. 冻结模型及来源

- 独立Pre_direct加载原Pre权重及冻结阈值/成本/回退，无重训。3个Baseline seed的7335个ID逐条匹配Task2正式动作/calls和Task3标签，两动作预测与q_hat在1e-9内一致，S431/L6904。
- WindowedSpatial.optimize与C盘final_release SpatialSimulation.optimize为同一函数，冻结5源码hash和实际模块路径匹配。仍使用30s每solve、原数值接受/恢复协议，无旧Full MPC替代。
- Difficult实得任务数General4401、Math2794、Code140，合计7335；从原历史任务重采样，不合成评分。Prediction Error实得原始误差均值/标准差/跨模型独立性检查通过，fallback不变。
- 增长保留原7335行和潜变量，追加3667唯一实例，总11002；只6host、不扩容。每个新动作的事前service与对应S/L冻结代理逐ID核对。

## 2. 世界、因果和窗口

- 固定seed={SEEDS}；每个world在执行前重建一次，完整payload SHA相同。每个情景/seed两策略world ID、完整route、误差、service潜变量、资源、availability、actual/forecast完全一致，12对通过。
- 历史质量仅DES run结束后连接。发生器/路由只用prompt、benchmark、事前服务/预测；没有用test q挑样本。原因果前缀/隐藏shock合同的实现文件未改，本轮solver与forecast audit逐次满足arrival<=decision、observed cutoff<=issued。
- H=43200，fixed idle不按最后完成缩短。每次N=completed+failed+censored，H时未完成=截尾；对应剩余工作保留。

## 3. 独立落盘复验

24条记录全部COMPLETE_VALIDATED，所有manifest工件hash和执行签名核对后，从CSV/JSON重新验证：ID终态、事件准入/replica容量、开始/完成/运行守恒、部分busy/controller区间、host一次idle、network不重复PUE、分项/总电量、独立逐小时碳积分、质量损害加性和互斥主结局。并验证最终Results每行总碳/总能量与原始metrics完全对应，非人为补齐数值。

本轮软件异常{len(failures)}，服务失败合计{int(results.failed.sum())}、截尾合计{int(results.censored.sum())}（跨策略执行计数，非独立请求数）。服务失败仍在分母和账本。未自然发生的窗口截尾不能据此宣称永不发生；原测试覆盖部分执行及H端点。所有旧失败日志未删。

独立验证：{str(checked.relative_to(HERE))}。保护/逐ID/场景合同：{str(protection_file.relative_to(HERE))}。新增适配diff在audit/runner_adapter.diff，可核对没有改变调度/核算语句。

## 4. 复用及资源

检查旧Task4的28条完整Pilot记录及Task3正式24种子记录；没有同N、同seed、同world的合法Baseline，故本轮新执行24次，没有重复同配置已有实验。旧历史记录不冒充独立新增样本。本轮重跑入口只有完全签名和工件验证相同时才引用旧记录，不增加独立样本。

运行前磁盘检查通过，保留2GiB研究预算及额外2GiB空闲底线；当前研究目录约{size/2**20:.2f}MiB，可用D盘约{shutil.disk_usage(HERE).free/2**30:.2f}GiB。实际完整执行计时合计{bench.execution_wall_s_before_final_manifest.sum():.2f}s（不含少量发生/最终manifest时间）；采样RSS最大{bench.sampled_peak_rss_bytes.max()/2**20:.2f}MiB，20ms采样不保证真实瞬态峰值。详细测量在audit/runtime_measurements.csv。

旧大规模57GiB以上预算仍未执行，本次约24执行的简化范围可容纳，不能据新实验规模解除旧C1–C8的资源限制。

## 5. 完整性与结论边界

原TASK4保护清单{protection['old_task4_files_checked']}文件无变化；D盘上游{protection['upstream_files_checked']}文件无变化；C盘final_release只读verify通过2467发布工件与11698既有保护文件。没有修改原权重、数据、配置、论文、结果或日志。

CSV逐seed结果、配对差、反弹和任务组表均由实际工件生成。仅三张图，无CVaR或LHS图。质量阈值0.5，P95取completed，质量/pass取全部选择，SLA与有效交付取全部输入；缺失组class标准化为NA。n=3仅描述均值与观察范围，不宣称统计显著、有限样本guard新覆盖、人口公平或部署测量。平均碳核算不等同电网因果边际减排，效率诱导需求未识别。

**本次简化任务完成，旧复杂实验停止。**输出没有未运行的实验数值；不继续扩大矩阵或改进优化器。
'''
    (HERE/'Task4_Simple_Validation.md').write_text(text,encoding='utf-8')
    for p in HERE.glob('*.md'):
        text = p.read_text(encoding='utf-8')

        text = re.sub(r'!\[([^\]]*)\]\((([A-Za-z]:/)[^\n]*?)\)',r'![\1](<\2>)',text)
        assert '\ufffd' not in text and text.count('```')%2==0
        for target in re.findall(r'\]\(<([A-Za-z]:/[^>]+)>\)',text):
            assert Path(target).exists(),target
        p.write_text(text,encoding='utf-8')
    figures = list((HERE/'figures').glob('*.png'))
    assert len(figures)==3
    for p in HERE.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    deliver = [HERE/'Task4_Simple_Results.csv',HERE/'Task4_Simple_Report_ZH.md',HERE/'Task4_Paper_Outline_EN.md',
               HERE/'Task4_Simple_Validation.md']+figures
    save(HERE/'audit'/f'delivery_manifest_{time.time_ns()}.json',{'status':'PASS','strategy_executions':24,
        'paired_worlds':12,'seeds_per_scenario':3,'C1_C8_executions':0,'training_executed':False,
        'all_results_trace_to_actual_executions':True,'deliverables':{str(p):sha(p) for p in deliver},
        'old_task4_changes':[],'upstream_changes':[]})
    print('PASS: 24 actual executions, 12 paired worlds, 3 figures, preserved upstream/legacy artifacts')


if __name__=='__main__':main()
