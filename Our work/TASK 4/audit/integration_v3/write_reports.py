from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from task4_core.sources import read,sha
from task4_core.serialization import save

HERE = ROOT/'audit/integration_v3'
RUN = ROOT/'runs/pilot_v3_001'
B = HERE/'benchmark_002'
r = read(B/'runtime_estimates_v3.json')
resource = read(HERE/'resource_snapshot.json')
execution = pd.read_csv(B/'execution_measurements.csv')
worlds = pd.read_csv(B/'world_measurements.csv')
scenario = pd.read_csv(B/'scenario_policy_measurements.csv')
baseline = read(HERE/'baseline_001/contract_results.json')
integration = read(HERE/'attempt_002/integration_results.json')
analysis = read(HERE/'analysis_001/results.json')
assert baseline['status']==integration['status']==analysis['status']=='PASS'
assert read(HERE/'protection_after_pilot.json')['status']=='PASS'
assert len(execution)==28 and len(worlds)==14


def link(path,label):
    return '['+label+'](<'+str(path.resolve()).replace('\\','/')+'>)'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def new(name,text):
    p = ROOT/name
    assert not p.exists(), 'Preserve previous delivery: '+name
    p.write_text(text.strip()+'\n',encoding='utf-8')


labels = {'normal':'正常','pre_single':'单因素Pre失准','load_single_fault':'到达压缩+单区故障',
          'hard_pre':'困难混合+Pre失准','carbon_reduced':'碳失准+长期容量下降',
          'common_fault_tail':'全部L故障+长尾','growth_hosts':'数量增长+新增物理host'}
pilot_rows = []
for name in labels:
    p = scenario[scenario.scenario.eq(name)].set_index('policy')
    w = worlds[worlds.scenario.eq(name)]
    pilot_rows.append([labels[name],int(p.loc['Greedy','n']),2,
        int(p.loc['Greedy','failed_sum']),int(p.loc['Spatial_MILP','failed_sum']),
        int(p.censored_sum.sum()),f'{w.pair_wall_s_including_generation_reuse_checks.mean():.3f}'])
pilot_table = table(['情景','每world N','seed数','P0服务失败合计','P1服务失败合计','两策略截尾合计','一对平均wall(s)'],pilot_rows)
native_calls = int(execution.native_highs_calls.sum())
optimization_calls = int(execution.solver_calls.sum())
max_rss = r['sampled_max_rss_bytes']/2**20
actual_mib = r['measured_world_and_execution_bytes']/2**20
evidence = f"""{link(HERE/'baseline_001/contract_results.json','27项基础回归')}；{link(HERE/'attempt_002/integration_results.json','10项集成合同')}；{link(HERE/'analysis_001/results.json','3项纯合成统计合同')}；{link(RUN/'summaries/1792336598054236000/pilot_summary.json','首次Pilot汇总')}；{link(RUN/'validations/1792336705483505800/results.json','28工件独立复验')}。"""

new('Task4_Pilot_Report_ZH.md',f"""
# TASK 4 受控 Pilot 报告 v3

日期：2026-10-18。**已实现并验证：受控工程Pilot；正式M1–M8/C1–C8执行数为0。**本报告不宣称系统优势、真实极端风险、反弹阈值或公平性已获验证。正式全量当前NO-GO的资源原因见Readiness v3。

## 1. 实际执行范围

先通过27项基础回归、10项新增集成合同，再运行7情景×2seed×2策略=28次新执行、14个外生世界。每世界800实例；增长情景从相同800基础cohort追加200个唯一实例，并把L host增加至各地区2台，共1000实例、9host。所有H=43200秒。seed为2028101800/1801，保留完整配置：{link(RUN/'pilot_config.json','Pilot配置')}。

同一seed跨情景使用公共随机子流；每情景有2个独立seed，14world不构成14个IID总体样本。P0/P1共享世界、路由、服务潜变量、碳场和容量；主模型是最终Spatial，不是旧Full。Pilot不占用正式MC/确认种子。

## 2. 完整性与实际观察

{pilot_table}

表中“失败”是DES服务终态，不是软件异常；故障情景有服务失败也必须保留。28/28均为COMPLETE_VALIDATED，逐ID满足N=completed+failed+censored；窗口、准入、队列生命周期、partial active/controller、host/PUE/network及独立碳积分通过。Pilot未自然产生窗口截尾；H附近运行/在途/排队/计算截尾由独立手工合同覆盖，不能声称Pilot实测覆盖了截尾频率。

共同L故障触发真实fallback路径；本Pilot没有产生MILP数值拒绝或solver timeout。数值拒绝→fallback仍交付的路径由基础合同注入测试覆盖，不能把注入例当自然发生率。历史质量只在每次DES结束后连接两动作score；任务复制仍按base/group追溯，独立质量样本数不增加。

没有将上述小样本差异用于调参或修改正式分布。没有对Pilot计算具有正式尾部支持含义的CVaR95，也没有用两seed认证任何部署预算。服务质量、环境效益和分组数值都只保留为本次模拟输出；部署判定仍为证据不足。

## 3. 异常、续跑与留存

首次Pilot没有软件失败或指标缺失。独立validate从文件重新检查28条记录全部通过。随后对同一配置执行--resume：28次全部为REUSED，新增DES=0；引用不增加独立样本。首次与续跑汇总分开保存，成本取首次汇总，不把复用耗时冒充新执行耗时。

集成测试另外有故意注入的异常和工件篡改，均保留在attempt_001/002的reuse_runner内。它们不属于Pilot失败，也没有从日志中删除。测试验证失败不可缓存为成功、重试新attempt、旧完整工件变化即拒绝复用。Pandas的可选numexpr/bottleneck版本警告仍保留，不修改全局环境。

## 4. 实测成本

首次串行Pilot含初始化约{r['pilot_wall_s']:.3f}s；一对world均值{r['world_pair_mean_s']:.3f}s，经验P95={r['world_pair_p95_s']:.3f}s。完整世界与28执行目录合计{actual_mib:.2f}MiB，不含另存的测试/交付文档。20ms进程RSS采样最大约{max_rss:.2f}MiB，包含模型/输入缓存，不能当精确瞬态峰值或全量内存上界。

Spatial日志共{optimization_calls}次优化调用，其中原生HiGHS {native_calls}次；其余含冻结实现的快速子问题路径。不能把每条优化日志都叫原生MILP。实测逐情景/策略wall、solver和文件大小见 {link(B/'scenario_policy_measurements.csv','分层测量表')}。

## 5. 结论边界

本轮完成工程连接与有界压力验证，没有完成正式MC尾部、公平性推断、完整LHS筛查或反弹根定位。完整7335压力可能出现更多批量MILP、数值反例和非线性成本；Pilot的最大wall不是上界。原完整IntegratedRouter仍环境阻断，正式所需Pre_direct独立路径可用。

{evidence}
""")

new('Task4_Integration_Validation.md',f"""
# TASK 4 端到端集成验证 v3

日期：2026-10-18。**40项不同合同通过：27基础+10集成+3纯合成统计。**重复执行合同不增加独立世界证据。另有14个Pilot world、28次新执行及28次严格复用引用；正式实验和训练均为0。

## 1. 接口与验证矩阵

{table(['接口/合同','实测验证','状态'],[
['Task2→world→Task3','新发生器完整7335逐ID动作、对应服务一致；S431/L6904','已实现并验证'],
['最终执行语义','冻结seed2027101800，7335新世界两策略对原引擎events/control完全一致；公共终态/账本/风险/总电量等atol1e-8','已实现并验证单世界'],
['预测误差','General随bias改变；Code/Math/unknown回退；新服务/能量/两动作q；truth poisoning不改路由','已实现并验证合同'],
['四类联合故障','高负载单区、困难Pre、碳误差容量下降、共同故障长尾；完整持久化后独立复验','已实现并验证160实例合同与800/1000 Pilot'],
['随机世界','同seed全payload相同、换seed不同；保存加载不变；策略顺序互换事件/终态一致','已实现并验证合同'],
['因果边界','仅_admit揭示Z；未来actual/容量变化不改前缀；相同整点共享误差；observed cutoff<=issued','已实现并验证合同'],
['增长/扩容','原实例保持、增长前缀嵌套、unique ID、同组同Pre误差、slots不增idle、新host增idle','已实现并验证合同'],
['窗口与失败','截尾部分busy/controller、未开始实际剩余未知；全输入风险/质量；异常留存且不能填0','已实现并验证合同'],
['续跑/哈希','注入失败后换attempt；篡改工件后拒绝复用；CLI同配置28条全复用无新DES','已实现并验证'],
['离线统计','2000条合成world标识×两策略，2000 Bootstrap验证已知平移；LHS恢复已知重要参数；反弹拒绝不同N0基线','已实现并验证合成合同，无系统风险结果'],
['完整7335压力/高增长/跨月','已具备基础实现；本轮未运行该正式矩阵','已实现但未充分验证'],
['原完整IntegratedRouter','unused scaler/state NumPy反序列化问题仍在；不影响独立Pre_direct','环境阻断']])}

## 2. 落盘后的独立验证

每条COMPLETE_VALIDATED必须保存execution/config/来源及环境版本、完整route、world映射与误差/实际场、requests/quality/failure/fairness、events/controls、issued forecasts与audit、solver/recovery、ledger和metrics。逐工件SHA和完整签名相符才可复用；验证不能仅信任status标签。

验证器重放事件中的容量/replica生命周期；容量下降时允许原有running继续，但任何新ServiceStart必须使用当前可准入空槽。独立计算host idle、所有已启动任务截至H的active、所有dispatch的network、计算区间与H交集，并按每小时实际碳积分核对总账。未完成任务保留在全输入风险与质量分母，不能用丢样本制造减排。

quality_harms保留signed和positive增量的各自加性；failure保存重叠机制及互斥主结局。稀疏公平组、W=0和缺class单元不填0；没有人口数据就不作人口公平声明。界限和因果测试不是对实际生产信息隔离的安全认证。

## 3. 来源与保护

始终继承C盘final_release SpatialSimulation.optimize，冻结source/hash检查拒绝旧模块混入。未修改优化方程、数值1e-5接受、恢复LP协议、30s预算或代理延迟；真实wall平移仍拒绝。TASK4只读上游，数值证据ROOT定向新attempt。

本轮D盘6402保护文件前后无变化；C盘只读--verify通过2467发布工件与11698既有保护文件。原始Task2/Task3模型、数据、论文、结果未写入。新增功能仅限TASK4；全部失败attempt保留。

## 4. 证据位置

{evidence}

集成主入口：{link(ROOT/'tests/run_integration.py','run_integration.py')}；基础：{link(ROOT/'tests/run_contracts.py','run_contracts.py')}；统计：{link(ROOT/'tests/run_analysis_contracts.py','run_analysis_contracts.py')}。测试日志保留在各attempt/analysis目录，不以本报告替代原始工件。
""")

cost_rows = [[e['module'],e['execution_references'],f"{e['linear_serial_hours']:.2f}",
              f"{e['planning_hours_2x_empirical_p95']:.2f}",f"{e['linear_disk_GiB']:.2f}",
              f"{e['disk_reserve_GiB_2x_p95']:.2f}"] for e in r['core_estimates']]
cost_rows.insert(4,['C5',0,'不增加DES','另计统计/IO','复用已有文件','不重复保存全世界'])
cost_rows.insert(6,['C7 Optional',192,'约1.21*','约2.50*','约2.37*','约5.01*'])
cost_table = table(['模块','执行引用','线性串行h','规划h','线性GiB','规划GiB'],cost_rows)
bench_rows = []
for row in scenario.itertuples():
    bench_rows.append([labels[row.scenario],row.policy,int(row.n),f'{row.simulation_mean_s:.3f}',
        f'{row.execution_mean_s:.3f}',f'{row.solver_calls_mean:g}',f'{row.native_highs_calls_mean:g}',
        f'{row.solver_wall_mean_s:.4f}',f'{row.files_mean_bytes/2**10:.1f}'])
benchmark_table = table(['情景','策略','N','DES均值s','完整执行均值s','优化调用均值','HiGHS均值','优化日志wall均值s','执行文件均值KiB'],bench_rows)
extended_table = table(['扩展','新增执行','线性h','规划h','线性GiB'],[
    [e['name'],e['additional_executions'],f"{e['linear_hours']:.2f}",f"{e['planning_hours_2x_p95']:.2f}",f"{e['linear_disk_GiB']:.2f}"] for e in r['extensions']])

runtime_text = f"""
# TASK 4 Runtime Benchmark / 成本估计 v3

日期：2026-10-18。**Pilot实测与正式计划估算分开；正式C1–C8未执行。**协议v2样本数/分布没有调整。以下成本在同机串行、完整世界/日志/验证条件下提取，不是硬件功率测量。

## 1. Pilot实测

{benchmark_table}

每行n=2执行，HiGHS次数的半数表示两个seed的均值，非半次求解。完整执行wall含来源复核、真实Pre推断、签名、world首次保存、DES、事后质量/公平、压缩落盘和独立验证；字段计时截至最终manifest前，world级pair wall还包括发生器和最终哈希，所以正式外推优先使用pair wall。DES纯事件循环远快于完整流水线，不能只乘DES时间估预算。

首次Pilot初始化{r['pilot_initialization_wall_s']:.3f}s，总计{r['pilot_wall_s']:.3f}s；14个world pair均值{r['world_pair_mean_s']:.3f}s、经验P95 {r['world_pair_p95_s']:.3f}s。Spatial优化调用总数{optimization_calls}，其中原生HiGHS {native_calls}；日志wall包括构造/快速子问题开销，不全是纯MILP CPU。

世界和执行工件实存{actual_mib:.2f}MiB；一实例在一个完整world及两策略下平均{r['bytes_per_instance_paired_mean']:.1f}bytes，P95 {r['bytes_per_instance_paired_p95']:.1f}bytes。world.json和压缩CSV当前同时保存，已计入；复用引用不再复制世界。独立验证/续跑及测试目录另计，未计为新Pilot执行。

进程RSS每20ms采样，实见最大{max_rss:.2f}MiB；16逻辑CPU、物理RAM约{resource['ram_total_bytes']/2**30:.2f}GiB，资源快照可用RAM约{resource['ram_available_bytes']/2**30:.2f}GiB。RSS包含持久输入/模型缓存，有采样遗漏可能，不能保证完整7335联合拥塞时的峰值。没有运行并行benchmark。

单区负载压力相对正常pair的实测额外开销和所有单world数据见 {link(B/'world_measurements.csv','world测量表')}；本小样本故障有时更快，因为失败减少执行，不可把它解释成服务提升。扩容增长及高批量原生HiGHS改变复杂度，不能统一按固定毫秒计。

## 2. 外推口径

每world先算 tau7335=tau_pair×7335/N，再取14world的均值{r['normalized_pair_mean_s_7335']:.3f}s、经验P95 {r['normalized_pair_p95_s_7335']:.3f}s。线性列=pair数×平均N倍率×均值；规划列=pair数×N倍率×2×经验P95。存储同样按每实例字节均值和2×P95外推。C4五r的平均N倍率约1.75。

这些是未验证的工程外推；两倍P95是资源安排系数，不是置信区间或严格上界。Pilot仅800/1000、每情景2seed，完整7335压力更可能形成批量MILP、恢复LP或反例NPZ；30s仅约束单solve，不约束world总wall。正式运行应逐world检查实际磁盘和耗时，超预算暂停记录而非删world/填0。

## 3. C1–C8估计（尚未执行）

{cost_table}

*C7不在Core：若未来按四月份×24seed×2=192引用执行，表中只是同N、同普通速度假设；新月份本轮未实现/未测。C5合成统计适配合同3项约{analysis['wall_s']:.2f}s，其中完整2000标识的合成CVaR适配用2000 Bootstrap；这是算法测时，不是2000个DES世界。正式多组CI、全文件读取另建议预留0.5–2h，尚未端到端实测，不加到“已测”时间。

C8未运行All-L Pilot，耗时仅借Greedy代理，规划用4倍该代理；存储保守沿用整对world额度。不能称All-L成本已经校准。正式计划共有4558引用；名义4540去重键仍必须经来源/配置/工件逐项验证，表中未预扣18次引用。

Core原始引用合计线性约{r['core_raw_reference_totals']['linear_serial_hours']:.2f}h、规划约{r['core_raw_reference_totals']['planning_hours_2x_empirical_p95']:.2f}h，加统计/IO；线性存储约{r['core_raw_reference_totals']['linear_disk_GiB']:.2f}GiB、建议至少{r['core_raw_reference_totals']['disk_reserve_GiB_2x_p95']:.2f}GiB，另留异常/Optional余量。既有历史normal全7335 pair约18.334s是旧生产IO口径；本轮完整流水线线性约45.431s，因此不能继续把旧11.8h当新完整执行预算。

## 4. Extended增量（只估计，不执行）

{extended_table}

M4边界表假设平均N=2N0；未知候选r或更高拥塞需重估。M6普通32点LHS不嵌套16点，因此512为新执行，不能扣192；8点补到12seed只有原4seed完全同键时才只新增128。M8扩展借整对world均摊代理，All-L负载未测。Optional CVaR99、真实wall、r4和更多主机另立预算，不纳入上述Core。

## 5. 存储阻断与不改变研究定义的修复选项

资源快照（{resource['observed_utc']}）显示D盘可用{resource['disk_free_GiB']:.2f}GiB，低于Core线性{r['core_raw_reference_totals']['linear_disk_GiB']:.2f}GiB，明显低于建议{r['core_raw_reference_totals']['disk_reserve_GiB_2x_p95']:.2f}GiB；C3单模块线性约49.35GiB已超当前空间。**当前正式全量NO-GO。**这不是降低CVaR独立world数量的理由。

可选下一步：提供至少约122GiB可用且明确授权的新结果卷；或另做存储格式v4，对world JSON无损压缩、公共prompt目录按hash引用、列式字典编码及完整性manifest做有界验证后重新测量。上述迁移/压缩尚未实施，不能把预计压缩率记为已节省空间。保留所有旧失败、当前Pilot和上游结果；不擅自清盘或外移数据。跨模块严格复用与延后Extended可以减小额外计算，不改变C3每层1000。

先串行执行、通过真实内存/磁盘门槛后再考虑有限并行；不按16核直接除时间。机器可读依据：{link(B/'runtime_estimates_v3.json','估算JSON')}、{link(B/'execution_measurements.csv','逐执行实测')}、{link(HERE/'resource_snapshot.json','资源快照')}。本文件不触发正式运行。
"""
new('Task4_Runtime_Benchmark.md',runtime_text)

new('Task4_Readiness_Report_v3.md',f"""
# TASK 4 Readiness Report v3

日期：2026-10-18。**NO-GO：当前不能启动正式Core全量实验，明确阻断为结果盘可用空间不足。**工程接口及受控联合Pilot通过；本轮到此停止，正式M1–M8执行数=0，训练次数=0。软件通过不等于假设或部署效果得到验证。

## 1. 已实现并验证

- Pre_direct只读真实推断；7335逐ID决策及两动作服务，S431/L6904；保留原fallback与原参数，不修改训练产物。
- 新ScenarioBuilder/ExogenousWorld：混合、group相关Pre误差、独立小时碳误差、流量压缩/burst/数量增长、单/多区故障恢复、长期容量下降、slots/多host及v2分布/LHS计划。
- 新流水线7335单个冻结世界的两策略对原最终引擎事件/控制/公共核算等价；固定窗口、部分active/controller、未完成工作与因果前缀合同通过。
- Task4Runner落盘、逐工件manifest、独立validate、异常部分轨迹留存、attempt重试与严格复用；28条CLI续跑全复用，无新增DES。
- 40项不同软件合同通过；14个Pilot世界、28次新执行全部COMPLETE_VALIDATED且独立复验通过。统计适配另用纯合成数据验证已知CVaR平移、LHS重要参数及反弹同N0保护，不产生系统尾部实验。

上述状态只覆盖明确样本和软件行为。来源依赖仍指向只读C盘final_release，缺失时停止，不回退旧模型。

## 2. 已实现但未充分验证

完整7335多seed联合压力、r最高2的22005实例、正式LHS15维完整闭环、All-L正常参照、原生HiGHS密集时的wall/RAM/反例文件量尚未实跑。世界与基础运行模块已实现，规模效果和统计结论未验证；不能将40合同或Pilot视为正式C1–C8完成。

M3风险适配器已在合成2000标识/2000 Bootstrap测试，但没有任何正式分层MC世界，因此真实所研究系统的CVaR/CI/tail ESS仍无正式结果。公平组与原因标签已逐world计算；正式world层推断、Holm检验族和独立确认仍必须按v2边界执行，不能把默认点态区间改称联合95%覆盖。

## 3. 仅设计

无损压缩/公共prompt目录的新存储版本、真实wall平移复查、异质pool host-aware优化、偏移增长、生产需求反馈及真实效率反弹识别仍仅设计。未实现Optional新月份的生产编排，也未采集新质量、实机功率/PUE、生产轨迹或人口属性。现有数据支持条件仿真，不支持真实部署认证。

## 4. 环境阻断

| 阻断 | 影响 | 下一步修复与验收 |
|---|---|---|
| D盘可用{resource['disk_free_GiB']:.2f}GiB；Core线性约57.42GiB，建议约121.43GiB，C3单模块约49.35GiB | 全量落盘/异常余量不足，可能生成不完整世界，禁止正式启动 | 获得至少约122GiB可用的授权结果位置；或实现并小测可审计无损存储版本后重估。不能删除失败/原数据或降低MC样本数来通过 |
| 原完整IntegratedRouter unused scaler/state的NumPy缓存不兼容 | 通用旧API仍不可构造；对本研究所用direct路径不构成阻断 | 已由独立Pre_direct绕开并全ID验证；不改旧joblib、不安装/重训上游 |

原生HiGHS线程/随机控制的不确定性继承冻结协议，未擅加不受支持选项；软件复现证据是当前同环境范围，不保证跨平台逐bit相同。

## 5. 放行条件与停点

唯一当前正式资源阻断是存储。消除阻断后，在同一模型/协议下复核：40合同和上游哈希、足够可用盘空间及新格式工件闭环（若变更）、完整签名/来源冻结、正式/确认与Pilot种子分离、失败不删且缺失不填0。正式开闸需要后续明确任务；当前CLI仅dry-run/pilot/validate，runner仍硬性拒绝超过1000实例的生产执行。这是本轮授权边界，未自动解锁。

不要求先用新数据证明现实模型才能做条件仿真，但输出必须明确历史重复cohort、假设分布、模拟origin/class、固定功率和缺少真实反弹因果数据的范围。部署资格仍需完整服务/质量/尾部/环境证据和代表性数据，Pilot不能据未来真实信息决定策略。

## 6. 留存证据

{evidence}

所有旧失败日志和新attempt_001/002均保留；上游D盘6402文件前后哈希无变化，C盘2467发布工件/11698旧保护文件只读验证通过。实测成本和剩余空间见 {link(ROOT/'Task4_Runtime_Benchmark.md','成本报告')}。正式协议v2的4558引用/名义4540独特键、M3两层各1000、LHS先2×16×3×2保持原定义。
""")


names = ['Task4_Implementation_Plan_ZH.md','Task4_Mathematical_Model_ZH.md','Task4_Execution_Checklist_ZH.md',
         'Task4_Review_Fixes_ZH.md','Task4_Core_Experiment_Protocol.md','Task4_Readiness_Report.md','Task4_Estimated_Runtime.md']
backup = HERE/'documents_v2'
backup.mkdir()
for name in names:
    shutil.copy2(ROOT/name,backup/name)
for name in names:
    if name=='Task4_Estimated_Runtime.md':
        (ROOT/name).write_text(runtime_text.strip()+'\n',encoding='utf-8')
    else:
        original = (ROOT/name).read_text(encoding='utf-8-sig')
        annotation = f"""> v3工程状态更新（2026-10-18）：联合世界和runner已实现，40项合同通过，14世界/28次受控Pilot及独立复验通过，续跑28条严格复用、无新DES。正式实验仍为0；当前因存储不足为NO-GO。最新状态见 {link(ROOT/'Task4_Readiness_Report_v3.md','Readiness v3')}，实测成本见 {link(ROOT/'Task4_Runtime_Benchmark.md','Runtime Benchmark')}。以下v2正文保留为设计/审计快照，其中“仅设计/尚无runner”等旧工程状态由v3报告覆盖；正式数学定义、分布和样本数不变。修改前完整副本在audit/integration_v3/documents_v2。

"""

        first,rest = original.split('\n',1)
        (ROOT/name).write_text(first+'\n\n'+annotation+rest.lstrip('\n'),encoding='utf-8')
save(HERE/'document_delivery_v3.json',{'status':'GENERATED_FROM_RETAINED_EVIDENCE','formal_experiments':0,
    'files':{str(ROOT/name):sha(ROOT/name) for name in names+['Task4_Pilot_Report_ZH.md','Task4_Integration_Validation.md',
       'Task4_Scenario_Generator_Spec.md','Task4_Runtime_Benchmark.md','Task4_Readiness_Report_v3.md']}})
print('Wrote v3 reports; preserved seven v2 originals; NO-GO storage gate retained')
