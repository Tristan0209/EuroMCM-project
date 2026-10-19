# Results and Discussion

We reran five preserved policies on all 7,335 frozen requests and 24 paired
service-noise seeds. Spatial MILP reduced facility emissions relative to
Greedy by 73.552477 g CO2e (paired 95% t interval
[73.229500, 73.875454]), and relative to the original
candidate-time Full MPC by 18.702516 g. These effects are conditional
on the fixed alignment, resource model and retrospective carbon snapshots;
they are not measured deployment savings.

The common idle base dominated facility electricity. Total percentage savings
were 0.178144%, whereas the corresponding fraction of Greedy incremental
emissions was 49.758542%. Reporting both avoids denominator ambiguity.
Failures and already-incurred transfer emissions remained in all-input totals.
Quality labels were used only for post-simulation delivered-utility accounting.

Fixed-integer continuous recovery did not alter assignments. The saved real
counterexample was genuinely infeasible after rounding, with a 0.0037348 s
bound contradiction, and was correctly rejected with explicit fallback.
Recoverable synthetic cases established software capability, not evidence
that all real rejected incumbents could be recovered. Nonzero holding in
the controlled E comparison cannot be interpreted as independent temporal
carbon value under flat persistence forecasts, particularly when it rescues
a failed zero-hold solve.

Safety-margin and measured-control experiments were rerun on the original
six-seed scope. Their results are exploratory and do not establish calibrated
chance constraints, zero violations or hardware-independent runtime. Native
fixed-work solver determinism, physical power calibration, operational as-of
carbon forecasts and cross-window generalization remain unverified.
