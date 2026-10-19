from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from task4_core.sources import read,sha
from task4_core.serialization import save
assert read(ROOT/'audit/integration_v3/additional_001/results.json')['status']=='PASS'
for p in ROOT.glob('*.md'):
    value = p.read_text(encoding='utf-8')
    value = value.replace('40项','42项').replace('40合同','42合同')
    value = value.replace('27基础+10集成+3纯合成统计','27基础+10集成+3纯合成统计+2补充合同')
    p.write_text(value,encoding='utf-8')

with (ROOT/'Task4_Integration_Validation.md').open('a',encoding='utf-8') as stream:
    stream.write('''
## 5. 收尾的两项补充合同

`tests/run_additional_contracts.py`两项通过（`audit/integration_v3/additional_001/`）：一项直接检查实际发生器的Pre组间S/L误差相关、均值1服务、64个小世界的小时碳误差相关及条件参数边际/故障概率；这些是固定seed的软件抽样检查，不是现实发生率估计，没有运行分布性能DES。另一项构造恰在H=43200发生Dispatch的单请求世界：离散network费用保留，active=0，inflight截尾，controller仅计窗内部分，全部验证通过。这一个端点toy不是新增Pilot或正式样本。

42项表示不同合同总数；反复回归和续跑不增加独立世界。`additional_001/distribution_checks.json`与`H_endpoint.json`保留原始检查数值。
''')
with (ROOT/'Task4_Review_Fixes_ZH.md').open('a',encoding='utf-8') as stream:
    stream.write('''
## v3 新发现与修复（2026-10-18）

| 问题 | 实际处理 | 状态 |
|---|---|---|
| 原ForecastView噪声键依赖精确issued | 新外生小时误差场，策略共享同一grid/region来源；未来信息前缀与随机场相关合同 | 已实现并验证注明范围 |
| 成本基线混用风险 | Pilot前发现草案增长情景N0=500而正常N0=800；改为同800基础cohort、r=.25追加200；反弹适配器拒绝不同N0。plan_001仅dry-run，实际用plan_002 | 已实现并验证；没有将草案记成实验 |
| 新route若沿用旧service会失真 | 按新动作取冻结train-only S/L代理，再乘明确倍率；质量仅仿真后连接两动作score | 已实现并验证 |
| 完整记录易把失败丢掉或被缓存 | RUNNING/FAILED/COMPLETE_VALIDATED、独立exception/partial文件、append-only台账、完整hash、失败保留和篡改拒绝 | 已实现并验证；28条真实Pilot续跑全复用 |
| H端点的网络离散费用记号不一致 | Core v2已规定同刻H事件处理/dispatch保留；单请求恰在H Dispatch合同通过。据既有协议将数学式4.14的严格小于号澄清为小于等于，连续功率仍[0,H)。没有改变执行策略/分布/采样数 | 已实现并验证；符号统一，不是新增假设 |
| 原normal历史时间不含新完整工件流程 | 改用14世界28执行含日志/验证的Pilot计时；区分优化日志调用与原生HiGHS，不把短DES耗时直接外推 | 实测完成；全量外推仍未验证 |
| 结果盘不足 | 实测可用33.87GiB，Core线性约57.42GiB/建议121.43GiB；正式全量NO-GO，提出新增授权存储或有界验证的新无损格式 | 环境阻断；尚未迁移/压缩，不删除旧日志或减少M3样本 |

最新状态见Readiness v3、Pilot和Runtime Benchmark；v2原文已在audit/integration_v3/documents_v2留存。正式M1–M8仍为0，LightGBM训练为0。
''')

p = ROOT/'Task4_Mathematical_Model_ZH.md'
value = p.read_text(encoding='utf-8')
assert 'dispatch_i<H' in value
value = value.replace('dispatch_i<H','dispatch_i\\le H')
value += '''
## v3 窗口端点记号统一（不改变既有协议）

Core Protocol v2及本说明第4节已规定处理t=H的同刻事件，并保留该时刻已dispatch的离散网络费用。式(4.14)的dispatch条件因此统一写成≤H；连续busy/controller仍按[0,H)交集。补充单请求合同验证dispatch恰等于43200时network>0、active=0、inflight截尾。本轮只是消除旧式严格小于号与既有文字规则的冲突，没有更改策略、随机分布或正式样本定义。
'''
p.write_text(value,encoding='utf-8')
with (ROOT/'Task4_Execution_Checklist_ZH.md').open('a',encoding='utf-8') as stream:
    stream.write('''
## v3 实施门槛更新

- [x] W01/W02：联合世界、独立子流、相关误差、条件参数分布及LHS计划实现并通过实际发生器抽样合同；不是现实率验证。
- [x] W03/W05：因果预测/容量/服务、group/instance映射、新动作服务及事后质量端到端合同。
- [x] W04/W06：生产工件、异常部分日志、完整manifest、严格复用和实际CLI续跑。
- [x] W07：14世界28执行有界联合Pilot，独立落盘复验通过；Pilot之外以toy覆盖H截尾/端点。
- [x] W08：Core v2分布/样本/损失不变；正式入口关闭，探索/确认种子分离。
- [ ] 正式资源门：D盘33.87GiB不足以容纳预计Core，需授权结果空间或经验证的新无损存储格式。
- [ ] 正式C1–C8：尚未执行；解除阻断不表示本轮获准自动运行。

合同总计42项（27+10+3+2）。v3状态以Readiness v3为准；上方v2勾选为历史快照。
''')

p = ROOT/'Task4_Estimated_Runtime.md'
value = p.read_text(encoding='utf-8').replace('# TASK 4 Runtime Benchmark / 成本估计 v3','# TASK 4 Estimated Runtime v3',1)
p.write_text(value,encoding='utf-8')
print('Finalized document annotations: 42 contracts, no new formal experiments')
