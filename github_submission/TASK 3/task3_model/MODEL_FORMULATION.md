# Pre-LightGBM + candidate-time MILP-MPC + event DES

## Scope and frozen upstream

The supplied Task 3 wording is the task source; the named competition PDF is absent. Model type m_i is immutable and inherited from Selected_finite_sample_guard (test, rep=0). Neither the optimizer nor DES receives historical true quality. S/L are anonymous resource services carrying historical Mixtral/GPT-4 routing labels, not controllable commercial API locations or measured hardware.

## Symbols

| Symbol | Meaning / unit |
|---|---|
| i in I_k | arrived, pending, uncommitted requests at decision k |
| r in R | DE, FR, ES |
| j in J_rm | admitted nonbatched GPU service slot on a fixed host (S/L separated) |
| u in U_irj | finite candidate starts / seconds from synthetic anchor |
| a_i, d_i | arrival and completion deadline / s |
| w_i | maximum combined intentional/resource waiting after mandatory routing/transfer / s |
| n_ir | one-way ingress + inherited network-control delay / s |
| ell_i | inherited router time / s |
| p_i | prior estimated service, already S/L scaled / s |
| pbar_i | safety planning service g*p_i / s, g configurable, not a guarantee |
| c_rm(t) | available new-admission replicas / integer |
| A_ir | allowed origin/destination and compatible-model indicator |
| C_r(t), Chat_r^k(t) | actual accounting and causal forecast / gCO2e/kWh |
| B_rmj(t) | already running/committed occupancy, immutable during a solve |
| x_irju | binary: start i on physical j at r at candidate u |
| z_i | binary unassigned by this restricted plan, explicitly handled by fallback |

## MILP

Candidates satisfy A_ir=1, u >= max(a_i+ell_i+n_ir, k+controller_delay+n_ir), u <= a_i+ell_i+n_ir+w_i, u+pbar_i<=d_i, u within planning horizon. A replica j is admissible for a new start only when j<c_rm(u). Capacity forecasts in deployable MPC persist current observed availability; future unannounced outage events are inaccessible. Oracle uses full event trajectory.

Assignment: sum_(r,j,u) x_irju + z_i = 1 for each i. m_i is fixed and no action changes it.

Nonpreemptive physical occupancy, for every elementary interval formed by all candidate/reservation start and end times:

    sum_(i,r,j,u: interval intersects [u,u+pbar_i)) x_irju <= 1 - B_rmj(interval).

These inequalities are per physical replica; overlapping jobs on one replica are forbidden. Already started tasks are not variables. Capacity loss blocks NEW starts on disabled replica indices, but grandfathered jobs continue; availability=0 does not imply power-off. The DES performs the same admission check using true current occupancy, and handles estimate overruns by queuing rather than double allocating.

Predicted incremental carbon for a candidate:

    e_irju = PUE_r*(Pbusy_r-Pidle_r)/3.6e6 * integral_u^(u+p_i) Chat_r^k(s) ds
             + transfer_kWh_ir * [Chat_origin^k(dispatch)+Chat_r^k(dispatch)]/2.

Fixed idle carbon is constant among candidate solutions in the common experiment window and is included in final accounting, never claimed as eliminated. Expected energy uses p_i; reservations use pbar_i. Carbon is integrated over every hourly boundary. Network accounting is a stated 50/50 endpoint attribution proxy.

The mathematical primary objective is min total carbon conditional on serving requests. When constraints cannot all be met, use a lexicographic objective: (1) minimize number unassigned, (2) minimize incremental carbon, (3) earliest start as tie-break. Implementation quantizes candidate carbon to micrograms. The unassignment coefficient exceeds the sum of all candidate maxima; normalized start ranks have total <1 carbon-quantum. This encodes priorities without arbitrarily adding seconds, dollars and grams. Quantization error is bounded by 0.5 microgram/request. z is not silent deletion: DES attempts earliest feasible fallback and records failure reason if it cannot serve.

Candidate reduction: earliest feasible start, a bounded set of subsequent service-spaced starts, known occupied endpoints, hourly boundaries and latest admissible start. Optimality is ONLY for this finite candidate set at the logged forecast, safety factor, batch and solver tolerance. Large Oracle runs use sequential offline MILP batches with shared reservations, not one global exact program; hence not a certified lower bound for the unrestricted stochastic system.

## MPC and forecasts

State contains arrived requests, observed queue/replica occupancy, announced commitments, current availability and past carbon observations. It excludes future arrivals, future service shocks, future actual carbon, and unannounced capacity changes. On arrival/deadline urgency or periodic tick, solve a bounded pending batch. Only starts inside the commitment interval are installed; farther plans remain uncommitted and are reconsidered. Running jobs cannot restart. Timeout/infeasibility/no incumbent is explicitly logged and falls back to earliest feasible compatible region. Singleton finite MILPs may be solved by exact enumeration, labelled separately from HiGHS calls.

Persistence uses only the last COMPLETE hourly observation, assumed available after one hour; forecast validity uses issued_at<=decision_time and observation_end<=issued_at. Historical-hour reforecast uses only past same-hour observations, falling back to past expanding mean. Simulated multiplicative forecast perturbations use fixed keyed seeds and no future actual values. These are causal retrospective reforecasts of a revised 2026 snapshot, not archived 2025 operational forecasts; as-of revision effects remain unverified. Perfect-foresight Oracle is explicitly separated.

## DES

A heap-based next-event simulator implements RequestArrival, RoutingDecision, Dispatch, ServiceStart, ServiceCompletion, CapacityChange, MPCReoptimization and DeadlineMiss (plus queue-ready/expiry/commit events). Physical time is continuous seconds. FCFS resource queues use arrival/ready sequence; policy intentional waiting is separate from waiting after transfer. Dispatch occurs after router/controller delay; service-ready=dispatch+one-way transfer+0.05s network control. Actual service is p_i times the common request/rep mean-one lognormal shock; only DES accesses it at ServiceStart.

intentional_delay = dispatch - (arrival+router_delay), including controller and policy holding. queue_wait = service_start - service_ready. response=completion-arrival. Thus response=router+intentional+network+queue+actual_service. Maximum allowed waiting bounds intentional+queue (with explicitly accounted mandatory overhead), while completion deadline independently limits completion. If prediction error causes overrun, started jobs finish and are marked late; unstarted impossible/expired jobs fail explicitly. All input IDs have exactly one terminal outcome. Completed historical quality is evaluated after simulation; fixed-cohort historical model quality remains invariant, with separate service-delivery failure metrics.

## Energy and carbon

For a shared observation window [0,H], all configured replicas incur idle energy regardless of chosen strategy or availability:

    Eidle_r = sum_(physical hosts h in r) (Pshared_h + G_h*Pgpu_idle_h) * PUE_h * H / 3.6e6
    Eactive_r = (Pgpu_busy_r-Pgpu_idle_r)*PUE_r*sum_i actual_service_ir / 3.6e6.

Total additionally includes network transfer kWh and separate incremental controller power (10W inherited, configurable). Per-request estimated_energy_kwh is only active increment, and L service must not be multiplied again. Carbon integrates each component over actual region/hour. Controller computation is modelled explicitly and wall solver times are separately recorded; measured runtime charging is a configurable sensitivity. Window H=43200 seconds is fixed across policies AND all demand/delay sensitivity levels, covering the preregistered load>=0.5 and delay_scale<=2 grid. It must cover all drain/completions, otherwise accounting fails rather than truncates. This prevents a changing deadline budget from artificially changing idle energy. Earlier pilot artifacts with shorter per-input windows are not used for final comparisons.

The current topology has six fixed 8-GPU DGX hosts, one S and one L host per DE/FR/ES region, and only one admitted service slot per pool. Installed GPU count (48) is not admitted capacity (6). Pshared=1500W/host, Pgpu_idle=175W/GPU, Pgpu_busy=760W/GPU, PUE=1.2. Thus each host consumes 2900W idle IT, with 585W additional IT per busy slot; total fixed idle facility energy is 250.56kWh/12h. p_idle_w/p_busy_w in server_config are compatibility aliases for 2900W and 3485W (host with ONE busy slot). Their difference in the MILP is exactly 585W; do not interpret 3485W as GPU power or multiply host idle by admitted slots. physical_host_config.csv and the pool-host map prevent double charging. Capacity loss never implies host power-off.

The profile is inherited from current Task1 and task2/runs/experiment and checked against its source core SHA256. Shared/idle power remains assumed; 760W is a specific public workload proxy, not a GPT-4/Mixtral or full-system measurement. The 14300W configured ceiling is a bound, not a mean. The old independent-replica 175/250W scenario is archived separately. Frozen request fields except prospective incremental energy remain unchanged; all new policy results require actual reoptimization/DES execution under the updated cost.

## Assumptions, scale and failure criteria

Input trace is synthetic Azure-shaped RouterBench replay translated to the earliest complete 2025 window after 24h history warmup; it is not observed 2025 business traffic. Lifecycle provider CO2e is primary; direct provider CO2e is separate sensitivity. Power, networking and hardware remain unmeasured assumptions. No actual gas-based GWP conversion is performed.

N arrivals and completions cost O(N log N) event-heap operations plus policy solves. A MILP with K candidates and L elementary occupancy constraints is NP-hard; sparse occupancy storage, bounded batches and solver deadlines limit cost, not prove unrestricted optimality. Report actual 100/500/1000/7335 results before scale claims. RQ1 compares Fixed Local vs spatial policies, RQ2 Greedy vs causal MPC with capacity/load changes, RQ3 variable delay budgets and Oracle headroom, RQ4 paired net carbon with SLA compliance, forecast error, spatial spread, network, congestion and computation sensitivity. Positive carbon saving alone never establishes practical worth when delivery constraints worsen.

## References

- SciPy 1.9.1 milp (HiGHS interface, statuses and limits): https://docs.scipy.org/doc/scipy-1.9.1/reference/generated/scipy.optimize.milp.html
- Electricity Maps lifecycle/direct and consumption accounting: https://www.electricitymaps.com/data/methodology . Snapshot-specific factors are not reconstructed from the current web page.

## 论文用约束的显式形式与保证边界

令候选索引集合为 \(\mathcal K_i=\{(r,j,u)\}\)。对不合法的地区/模型、尚未到达、网络未就绪、容量不允许准入、超过最大等待、预计无法按时完成的候选，等价地固定 \(x_{irju}=0\)。保留候选须满足：

\[
\begin{aligned}
x_{irju}&\in\{0,1\},\quad z_i\in\{0,1\},\\
\sum_{(r,j,u)\in\mathcal K_i}x_{irju}+z_i&=1,\\
x_{irju}&\le A_{ir},\qquad j\in J_{r,m_i},\\
u&\ge a_i+\ell_i+n_{ir},\\
u&\ge k+\tau_{\mathrm{controller}}+n_{ir},\\
u-(a_i+\ell_i+n_{ir})&\le w_i,\\
u+\bar p_i&\le d_i,\qquad u+\bar p_i\le k+H_{\mathrm{MPC}},\\
j&<\widehat c^k_{r,m_i}(u)\quad\text{(new-admission eligibility)}.
\end{aligned}
\]

上述关于u的关系是**生成变量前的候选过滤**，不会把未选候选的u再当成连续变量约束所有请求。对由全部候选/固定占用端点形成的基本区间 \(b=[b_0,b_1)\)，完整非抢占占用为：

\[
\sum_{i,u:\,[u,u+\bar p_i)\cap b\ne\varnothing}
x_{irju}\le 1-B_{r,m_i,j}(b).
\]

\(B\)是既有占用区间并集的指示量，运行任务不成为新变量。可用容量下降以副本索引屏蔽新启动，已启动区间继续占用其物理副本；因此瞬时running总数可能暂高于新的准入容量，但永不高于配置物理副本数，且被停用副本不接新任务。当前主配置每池一个副本，零容量时只可能出现一个历史在服任务继续完成。

碳量化后的实现目标可写为：

\[
\min\; P\sum_i z_i+\sum_{i,r,j,u}
\left[\operatorname{round}(10^6 e_{irju})+
\frac{\mathrm{rank}_{irju}}{|I|(K+1)}\right]x_{irju},
\quad
P=2+\sum_i\max_{r,j,u}\operatorname{round}(10^6e_{irju}).
\]

第二项是微克CO2e量化值，最后的无量纲rank总贡献小于一个碳量化单位，只作数值平局处理；没有把美元、秒、克任意相加。该支配系数对应精确求解时的字典序解释；实际HiGHS有记录的数值/相对gap容差，status=0仍表示求解器容差内最优，不能声称精确数学最优或任意连续候选的最优。未分配状态必须进入DES回退/失败账本。

均值服务代入的 \(e_{irju}\) 是预测近似；若服务跨小时且随机长度不确定，它不等于随机积分期望的精确解析式。DES使用实际实现长度逐小时核算，故评价不依赖这个近似成为精确值。

**为什么不能承诺随机零违反：** 继承的扰动 \(Z\sim\operatorname{Lognormal}(-\sigma^2/2,\sigma^2)\) 有无界支撑。即使不排队，对任何有限正可用服务预算L都有

\[
\Pr(p_iZ>L)=1-\Phi\!\left(\frac{\log(L/p_i)+\sigma^2/2}{\sigma}\right)>0.
\]

因此规划层的硬截止仅相对于 \(\bar p_i=1.35p_i\)；它不是对真实随机执行的绝对保证。要满足业务概率SLA，须另行预先定义并校准风险预算、准入/拒绝规则和尾部模型；本次没有根据实验失败事后调大安全余量来掩盖原结果。所有随机违反/失败仍保留，严格“值得”判定拒绝这些样本情景。
