# Final model: frozen routing, spatial MILP and A2 DES

## 1. Scope and information

The upstream Selected_finite_sample_guard Pre-LightGBM router is frozen. The
7335 request IDs, 431 S / 6904 L choices, estimated service times and historical
quality are immutable. Code/Math and unregistered tasks retain L fallback.
S/L are abstract services with Task 2 historical quality, not a claim of control
over GPT-4 server geography or measured hardware electricity.

At decision time s, only arrived pending requests I_s, known reservations,
current observed admission capacities, and forecasts issued no later than s
are available. Evaluation actual carbon, future arrivals, future outages and
unrevealed service shocks are excluded. Quality labels enter evaluation only
after simulation ends. A2 DES is byte-identical to the audited predecessor.

## 2. Symbols

| Symbol | Meaning / unit |
|---|---|
| i, r, m, j | request, DE/FR/ES region, frozen S/L service, physical admission slot |
| a_i, D_i, W_i | arrival, completion deadline, extra-wait limit, seconds |
| p_i | frozen service estimate, seconds; already includes S=1, L=3 |
| gamma | nominal reservation safety multiplier; baseline 1.35 |
| n_ir, E_ir^net | frozen network delay seconds and transfer kWh |
| b_rmj | latest end of known nominal reservations on slot j |
| C_hat_s(t,r) | causally issued forecast, life-cycle gCO2e/kWh |
| C_actual(t,r) | actual snapshot, evaluation environment only |
| y_irj, z_i | binary assignment and explicit unassigned state |
| u_irj | continuous nominal queue start, seconds relative to aligned UTC anchor |
| H | common energy/accounting horizon, 43200 seconds |
| d | optional whole-batch holding, {0,30,120,240} seconds for E comparison |

## 3. Spatial MILP

The model retains only type-compatible, network-allowed, currently available
slots. With router overhead h=0.01 and nominal optimizer overhead tau=0.01:

\[
R_{ir}(d)=s+\tau+d+n_{ir},\quad
e_{irj}(d)=\max(R_{ir}(d),b_{rm_i j}),\quad
\ell_{ir}=\min(D_i-\gamma p_i,a_i+h+n_{ir}+W_i).
\]

Choices with e>ell are excluded. Each request is assigned once or explicitly
unassigned:

\[
\sum_{rj}y_{irj}+z_i=1,\quad y,z\in\{0,1\},\quad
0\le u_{irj}\le H,\quad e_{irj}y_{irj}\le u_{irj}\le\ell_{ir}y_{irj}.
\]

On each slot, candidate order is fixed by (network delay, arrival, ID). For
candidate i preceding k:

\[
u_{irj}-u_{krj}+M y_{irj}+M y_{krj}\le2M-\gamma p_i,
\quad M=H+\max_i\gamma p_i+1.
\]

Both selected implies complete, nonpreemptive nominal occupancy. Existing
reservations cannot be rescheduled. Current capacity is persisted, not
clairvoyantly projected. There are six physical hosts and six baseline
admission slots; 48 installed GPUs do not imply 48 available service slots.

The expected incremental-carbon proxy for an assignment is

\[
c_{irj}(d)=\frac{1.2\cdot585\,p_i}{3.6\times10^6}\widehat C_s(s+d,r)
+\frac{E_{ir}^{net}}2[\widehat C_s(s+d,o_i)+\widehat C_s(s+d,r)].
\]

The implemented objective is

\[
\min\sum_{irj}\operatorname{round}(10^6 c_{irj})y_{irj}
+P\sum_i z_i+\epsilon\sum_{irj}u_{irj},
\]

where P is the sum of per-request maximum quantized costs plus 2, and
epsilon=1/((K+1)(H+1)). This is a bounded priority construction: maximize
assignments, then microgram-quantized carbon, then an early-time tie-break.
It does not arbitrarily add seconds, dollars and grams. Finite precision and
the solver's MIP termination criteria limit exact lexicographic claims.
The proxy uses expected durations and point forecast values; actual carbon
uses exact interval integration, so these are distinct quantities.

Spatial dispatches immediately after computation, with d=0. u certifies
nominal assignment feasibility; actual start is determined by A2's pooled
FCFS admission. Existing in-flight work is not completely represented by
the inherited nominal reservation approximation. Neither gamma nor MILP
feasibility guarantees actual SLA compliance under uncertain service.

## 4. Original Full MPC and E controls

Original Full MPC retains its finite request/region/slot/start candidates
x_irtj. A request has one chosen start or unassigned state. For every
elementary interval between candidate starts and ends, the sum of selected
occupancies on a slot is at most one. Candidate starts must satisfy arrival,
network, maximum waiting and completion deadline with duration gamma p_i.
The expected active-carbon cost is integrated over predicted p_i. Candidates
include earliest feasible start, finite service-spaced alternatives, latest
start, known reservation ends and hour boundaries. The candidate-step count
is five, planning horizon 10800 seconds, maximum arrived batch 64, and
reoptimization interval 300 seconds. This is candidate-restricted online
optimization, not an unrestricted global optimum.

E-Spatial and E-Full compute the same four fixed-d Spatial MILPs using equal
7.5-second per-branch budgets. E-Spatial can select only d=0. E-Full selects
among successful branches by (unassigned count, quantized carbon, d).
The zero branch is explicitly included. With persistence forecasts and
persisted capacities, raising d changes only earliest-start bounds; hence
F(d) is a subset of F(0) in nominal (y,u) space. Nonzero d cannot improve a
fully solved, identical-state nominal objective in that setting. Numerical
rejections or inconsistent solving can nevertheless trigger different
branch choices. Such rescue is not evidence of independent temporal carbon
value. Full controller actions including fallback are not proved nested.

Fixed Local, Greedy and Offline Oracle remain in the versioned A2 engine.
Oracle may use future actuals only under its explicit Oracle label; finite
batch/candidate Oracle is not a theoretical unrestricted optimum. The final
five-policy rerun does not constitute a new Oracle or Fixed Local experiment.

## 5. Full mixed-integer solution recovery

The raw result must have status 0. For integer set J, fix
v_J=round(v_J^raw). Reject if raw integrality distance exceeds the inherited
1e-5 threshold. Validate the complete rounded vector against every original
bound and row. If valid, retain the original continuous values.

If invalid and continuous variables C exist, solve

\[
\min c_C^\top v_C,\quad
L-A_J\bar v_J\le A_C v_C\le U-A_J\bar v_J,\quad
l_C\le v_C\le h_C.
\]

Only positive objective rescaling is used, preserving the fixed-integer
secondary objective. No integer assignment or refusal decision is changed.
The LP receives only the original subproblem's remaining wall budget;
primal/dual tolerances are 1e-7. All original rows, variable bounds and
integrality are then checked again at the inherited absolute 1e-5 threshold,
followed by nominal arrival/deadline/order/type/allowed-slot checks. An LP
timeout, infeasibility, invalid restored vector or lack of continuous freedom
is rejected as status4 and handled by the existing explicit fallback.

Pure-binary candidate Full MPC has no continuous degrees of freedom. Its
rounded vector receives full bound/row validation; an invalid assignment is
rejected. No alternate integer solution is invented. Recovery does not
establish an exact global-optimality certificate for the mathematical model.

## 6. A2 event-driven execution

Same-time order is CapacityChange, ServiceCompletion, QueueReady,
RequestArrival, RoutingDecision, MPC/Commit, Dispatch, Admission, Expire,
DeadlineMiss. Coalesced admission executes after same-time capacity updates.
Running work continues after a capacity reduction; new admissions must be
below the updated admissible capacity. No preemption, restarting or duplicate
execution is introduced.

The serial controller considers only arrived pending work. It commits plans
within the rolling interval; longer plans remain pending. Network readiness
is dispatch+n_ir. Resource queue wait is service_start-ready. Intentional
delay in the inherited output includes batching, controller waiting and any
explicit holding; it is not synonymous with voluntary carbon holding.
Response is completion-arrival. Actual duration p_i Z_i is disclosed only
on ServiceStart, with common mean-one lognormal shocks sigma=0.15 shared
across paired policies. Every request receives a completed or failed record;
completion after deadline is counted as a violation, not silently removed.

H measures each new optimization's local elapsed wall w, setting controller
duration L=0.01+k w. After computation, plans shift by k w and are checked
against nominal deadlines/wait limits again. Invalid commitments go to the
same explicit fallback. This is a closed-loop feasibility heuristic, not
optimization with advance knowledge of future CPU duration. H is run
sequentially without other experiment workers. CPU queue waiting is not
charged again as CPU execution electricity.

## 7. Energy and carbon

Each DGX host has shared idle 1500W plus eight installed GPU idles of 175W:
2900W once per host. Each actually busy GPU adds 760-175=585W. PUE is 1.2.
Unavailability does not power a host off. Six hosts over the common 12 hours
contribute exactly 250.56kWh of idle energy for every policy.

\[
E=\sum_h\frac{1.2\,2900H}{3.6\times10^6}
+\sum_i\frac{1.2\,585\,p_iZ_i}{3.6\times10^6}
+E^{net}+E^{controller}.
\]

The request-level estimated_energy_kwh is busy-over-idle only. Do not add a
second full busy power term or multiply frozen L durations by three again.
Network electricity is split equally between origin/destination at dispatch.
Controller electricity is incremental 10W times PUE times actual modeled
computing duration, separate from DGX idle/active accounting.

\[
CO2e=\sum_r\int_0^H\frac{P_r(t)}{3.6\times10^6}C_{actual}(t,r)dt
+CO2e_{network}.
\]

Service and control intervals crossing hours are split/integrated exactly.
The data's life-cycle gCO2e/kWh convention is retained; it is not relabeled
as direct gCO2/kWh. Actual carbon is retrospective evaluation, not an
operational as-of forecast. The Jan 2 2025 anchor is fixed before outcomes;
RouterBench requests retain their synthetic Azure-shaped arrival process.

## 8. Statistical and computational limits

Core comparisons use 24 independent service seeds and common random numbers
within each seed. Seed trajectories, not requests or MILP calls, are the
units of paired Student-t confidence intervals. F/H use the original six-seed
scope and are exploratory. These intervals are conditional on one aligned
window, frozen requests and the assumed resource model. Gamma sensitivity
does not calibrate a chance constraint or establish zero-risk service.

With N<=64 and K compatible slot choices, Spatial has 2K+N variables and
O(N+K+sum_pool K_pool^2) rows. Original Full depends on retained candidate
starts and elementary occupancy intervals. MPC is bounded by arrived batches,
not one monolithic second-resolution 7335-request optimization. Solver-budget
termination and fallback remain part of the evaluated policy. Native fixed
work/thread/seed control is unverified in SciPy1.9.1/HiGHS1.2.2. Installed
GPU count, admission slots and execution utilization remain distinct. No
simulation result is claimed as measured real-deployment emissions savings.
