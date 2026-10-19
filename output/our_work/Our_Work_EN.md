# 1.3 Our Work

We develop a quality-aware routing and carbon-aware scheduling framework for SWITCH. Figure X summarizes four linked stages of our work.

First, we combine public benchmarks, workload traces, energy evidence and grid-carbon data to evaluate response quality, inference cost, latency and emissions. Pre-inference quality prediction and post-response assessment provide complementary evidence for comparing routing policies.

Second, we compare direct and sequential-cascade policies under quality and cost constraints. The selected Pre-LightGBM router uses task-specific finite-sample risk guards and fallback to the more capable model; post-response escalation is not activated in the selected policy.

Third, we keep model choices fixed and formulate regional allocation as a Spatial MILP, with execution evaluated through discrete-event simulation. Comparisons with Greedy and temporal scheduling alternatives examine the trade-offs among carbon emissions, tail latency and service violations.

Finally, we assess robustness under task-mix shifts, prediction noise and demand growth, and examine fairness across request categories, rebound effects and total facility emissions. Together with sensitivity and overhead analyses, these results identify the conditions under which additional routing or scheduling complexity is worthwhile and inform recommendations to SWITCH.

**Figure caption:** Overview of our framework: performance evaluation, protected model routing, carbon-aware scheduling, and robustness and sustainability assessment.

## Compact version for a tight introduction page

We organize our work into four linked stages (Figure X). First, we establish a common framework for quality estimation and cost, latency and carbon accounting. Second, we compare direct and cascade policies and select a Pre-LightGBM router protected by task-specific risk guards and fallback. Third, with model choices fixed, Spatial MILP and discrete-event simulation evaluate regional allocation against Greedy and temporal scheduling alternatives. Finally, stress tests, task-level fairness assessment and rebound analysis examine service risks and total environmental impact. Sensitivity and overhead analyses help identify when additional complexity is worthwhile and support recommendations to SWITCH.
