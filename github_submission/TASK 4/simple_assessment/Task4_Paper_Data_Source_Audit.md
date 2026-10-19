# Task 4 Paper Data Source Audit

Status: PASS for the numerical and mathematical checks below. This writing pass reuses the original 24 executions and the final review statistics; it performs no training, model inference or simulation. All 5,175 rows of Additional Analysis and all 24 Results rows were read. All manuscript percentages are seed-level ratios averaged over three seeds; no confidence interval or significance claim is introduced.

## 1. Numerical traceability

The machine-readable audit is `paper_writing/paper_audit.json`. Source-table names in the following table are relative to this directory. The same values are indexed by analysis/scenario/seed/policy/task/region/metric in [Additional Analysis](Task4_Additional_Analysis.csv). Support tables already trace to each immutable run manifest, quality_harms and energy ledger through [Final Validation](Task4_Final_Validation.md) and `final_review/calculation_audit.json`.

| Paper claim | Display | Unit | Saved source / computation | Comparison |
| --- | --- | --- | --- | --- |
| Baseline Spatial p95_response_s | 7.827 | s | Task4_Simple_Results.csv :: p95_response_s | mean of three seed-level Spatial values |
| Difficult Spatial p95_response_s | 11.967 | s | Task4_Simple_Results.csv :: p95_response_s | mean of three seed-level Spatial values |
| Demand Growth Spatial p95_response_s | 11.885 | s | Task4_Simple_Results.csv :: p95_response_s | mean of three seed-level Spatial values |
| Baseline Spatial service_violation_rate | 0.0182 | % | Task4_Simple_Results.csv :: service_violation_rate | mean of three seed-level Spatial values |
| Difficult Spatial service_violation_rate | 0.0182 | % | Task4_Simple_Results.csv :: service_violation_rate | mean of three seed-level Spatial values |
| Demand Growth Spatial service_violation_rate | 0.1697 | % | Task4_Simple_Results.csv :: service_violation_rate | mean of three seed-level Spatial values |
| Demand Growth Spatial completion_rate | 99.8303 | % | Task4_Simple_Results.csv :: completion_rate | mean of three seed-level Spatial values |
| Baseline Spatial mean_historical_quality | 0.776142 | Score | Task4_Simple_Results.csv :: mean_historical_quality | mean of three seed-level Spatial values |
| Difficult Spatial mean_historical_quality | 0.748728 | Score | Task4_Simple_Results.csv :: mean_historical_quality | mean of three seed-level Spatial values |
| Baseline Spatial answer_pass_rate | 83.817 | % | Task4_Simple_Results.csv :: answer_pass_rate | mean of three seed-level Spatial values |
| Difficult Spatial answer_pass_rate | 84.872 | % | Task4_Simple_Results.csv :: answer_pass_rate | mean of three seed-level Spatial values |
| Growth Greedy service_violation_rate | 1.6270 | % | Task4_Simple_Results.csv :: service_violation_rate | Growth Greedy only |
| Growth Greedy p95_response_s | 6.998 | s | Task4_Simple_Results.csv :: p95_response_s | Growth Greedy only |
| Baseline Math router_quality | 0.6624 | Score | final_review/router_vs_AllL.csv :: router_quality | historical routing quality only |
| Baseline General router_quality | 0.8125 | Score | final_review/router_vs_AllL.csv :: router_quality | historical routing quality only |
| Baseline Math router_pass | 91.99 | % | final_review/router_vs_AllL.csv :: router_pass | historical routing quality only |
| Baseline General router_pass | 81.51 | % | final_review/router_vs_AllL.csv :: router_pass | historical routing quality only |
| L_to_S | 67/75/68 | requests per seed | final_review/prediction_flips.csv :: L_to_S | same changed base IDs; seed order 2029101800/1801/1802 |
| S_to_L | 70/64/65 | requests per seed | final_review/prediction_flips.csv :: S_to_L | same changed base IDs; seed order 2029101800/1801/1802 |
| material_drop_count | 19/22/18 | requests per seed | final_review/prediction_flips.csv :: material_drop_count | same changed base IDs; seed order 2029101800/1801/1802 |
| lost_L_pass_due_to_new_S | 13/15/12 | requests per seed | final_review/prediction_flips.csv :: lost_L_pass_due_to_new_S | same changed base IDs; seed order 2029101800/1801/1802 |
| Switches with equal scores | 71.63 | % | final_review/prediction_flips.csv :: quality_unchanged_count/n_changed | mean of seed ratios |
| Baseline total Greedy | 41,288.178 | gCO2e | Task4_Simple_Results.csv :: co2e_g | same baseline demand; all facility components |
| Baseline total Spatial_MILP | 41,214.556 | gCO2e | Task4_Simple_Results.csv :: co2e_g | same baseline demand; all facility components |
| Baseline same-demand carbon saving | 73.622 | gCO2e | final_review/energy_paired.csv :: carbon_saving_g | Greedy minus Spatial within baseline world |
| Baseline facility saving rate | 0.1783 | % | final_review/energy_paired.csv :: carbon_saving_fraction | mean (G-S)/G; full facility denominator |
| Fixed idle carbon | 41,140.3512 | gCO2e | final_review/energy_comparison.csv :: idle_carbon_g | common six-host 12-hour baseline |
| Idle share | 99.82 | % | final_review/energy_comparison.csv :: idle_carbon_fraction | mean idle / Spatial total |
| Spatial extra baseline electricity | 0.004857 | kWh | final_review/energy_paired.csv :: -energy_saving_kwh | Spatial minus Greedy, same baseline world |
| Growth matched-demand saving | 101.443 | gCO2e | final_review/energy_paired.csv :: carbon_saving_g | Growth Greedy minus Growth Spatial |
| Cross-demand change | +2.496 | gCO2e | final_review/rebound_audit.csv :: cross_demand_delta_g | Growth Spatial minus Baseline Greedy |
| Growth Spatial total | 41,290.674 | gCO2e | Task4_Simple_Results.csv :: co2e_g | Growth Spatial facility total |
| Cross-demand relative change | 0.006045 | % | final_review/rebound_audit.csv :: cross_demand_delta_fraction | mean seed change / original Greedy total |
| Linear reference | 99.21 | % growth | final_review/rebound_audit.csv :: r_linear_reference | fixed idle and constant per-input operating coefficient |
| Baseline FR active energy share | 89.583 | % | final_review/region_execution.csv :: active_energy_share | FR share of active-increment electricity only |
| Baseline active weighted intensity | 42.237 | gCO2e/kWh | final_review/energy_comparison.csv :: active_effective_intensity | mean seed active carbon / active kWh |
| Demand Growth FR active energy share | 79.632 | % | final_review/region_execution.csv :: active_energy_share | FR share of active-increment electricity only |
| Demand Growth active weighted intensity | 57.231 | gCO2e/kWh | final_review/energy_comparison.csv :: active_effective_intensity | mean seed active carbon / active kWh |
| General offline quality router_quality | 0.812466 | Score | final_review/router_vs_AllL.csv :: router_quality | paired Router minus offline All-L; not DES All-L |
| General offline quality AllL_quality | 0.818424 | Score | final_review/router_vs_AllL.csv :: AllL_quality | paired Router minus offline All-L; not DES All-L |
| General offline quality delta_quality | -0.005958 | Score | final_review/router_vs_AllL.csv :: delta_quality | paired Router minus offline All-L; not DES All-L |
| General offline quality delta_pass | -0.5055 | percentage points | final_review/router_vs_AllL.csv :: delta_pass | paired Router minus offline All-L; not DES All-L |
| Code n_input | 86 | count | final_review/router_vs_AllL.csv :: n_input | original baseline; distinct groups are not proven independent |
| Code unique_prompt_groups | 85 | count | final_review/router_vs_AllL.csv :: unique_prompt_groups | original baseline; distinct groups are not proven independent |
| Growth General violation_rate | 1.5925/0.1404 | % | final_review/group_service.csv :: violation_rate | three seed means; Greedy/Spatial, same task and world |
| Growth General completion_rate | 98.408/99.860 | % | final_review/group_service.csv :: completion_rate | three seed means; Greedy/Spatial, same task and world |
| Growth General completed_p95_s | 4.896/10.502 | s | final_review/group_service.csv :: completed_p95_s | three seed means; Greedy/Spatial, same task and world |
| Growth General valid_pass_rate | 80.247/81.438 | % | final_review/group_service.csv :: valid_pass_rate | three seed means; Greedy/Spatial, same task and world |
| Growth Math violation_rate | 1.7354/0.2482 | % | final_review/group_service.csv :: violation_rate | three seed means; Greedy/Spatial, same task and world |
| Growth Math completion_rate | 98.265/99.752 | % | final_review/group_service.csv :: completion_rate | three seed means; Greedy/Spatial, same task and world |
| Growth Math completed_p95_s | 8.493/13.945 | s | final_review/group_service.csv :: completed_p95_s | three seed means; Greedy/Spatial, same task and world |
| Growth Math valid_pass_rate | 90.293/91.703 | % | final_review/group_service.csv :: valid_pass_rate | three seed means; Greedy/Spatial, same task and world |
| Growth Code violation_rate | 1.7590/0.4919 | % | final_review/group_service.csv :: violation_rate | three seed means; Greedy/Spatial, same task and world |
| Growth Code completion_rate | 98.241/99.508 | % | final_review/group_service.csv :: completion_rate | three seed means; Greedy/Spatial, same task and world |
| Growth Code completed_p95_s | 12.211/18.329 | s | final_review/group_service.csv :: completed_p95_s | three seed means; Greedy/Spatial, same task and world |
| Growth Code valid_pass_rate | 67.435/68.204 | % | final_review/group_service.csv :: valid_pass_rate | three seed means; Greedy/Spatial, same task and world |

The 133-139 flip range is min/max of n_changed across the three seeds, with all flips General and zero Math/Code. All individual declines observed here exceed the declared 0.10 reporting threshold. L-pass loss means old L/new S with q_L>=0.5 and q_S<0.5; these counts are not net new service failures. Approximately 71.63% is the mean of seed-wise tie proportions, not a pooled proportion.

The three growth rows of Table 1 are independently reconstructed from group_service. The 36 policy contrasts are 4 scenarios x 3 seeds x 3 task categories; zero have a positive Spatial-minus-Greedy violation difference. They are descriptive comparisons, not 36 new independent quality samples.

## 2. Design constants and prior chapters

| Manuscript item | Authoritative source and interpretation |
| --- | --- |
| 7,335 baseline / 11,002 growth requests | protocol.json; growth appends floor(0.5 x 7,335)=3,667 instances with retained base IDs/groups; actual growth 49.9932% |
| 40% Math/Code | protocol.json Difficult.hard_fraction=.4; composition proxy, not measured item difficulty |
| Gaussian SD 0.10 / clipping [0,1] | protocol.json prediction_noise; per-group independent action errors, saved quality predictions perturbed before frozen decision recomputation |
| Six hosts / 12-hour H=43,200 s | Final Chinese report section 1; protocol; Task3 equations (3.4)-(3.6); identical fleet and horizon across policies |
| Three seeds / 24 executions | protocol seeds 2029101800/1801/1802; four scenarios x two policies; paired world_id verified in Final Validation |
| Score threshold 0.5 | Task 2 evaluation convention and final_review/statistics_schema.json; not an official real-world correctness standard |
| Frozen Pre-LightGBM + guard | ../../task2/runs/experiment/final_model_choice.json, final_safety_calibration.json, selected_parameters.json; family Pre_direct/A_Baseline, General lambda=.5, tau=-.617099253690552; Code/Math/unregistered fallback L |
| Frozen decision, no inference in this pass | ../../task2/integrated_api.py and ../task4_core/router.py: score=(qhat_L-qhat_S)-lambda*normalized_cost_delta; choose L iff score>=tau for registered General |
| Spatial MILP + DES | ../../TASK 3/task3_model/task3_final_paper/Task3_Paper_EN.md and Task3_Equations.md; final-release provenance already recorded in each original manifest; no original Full MPC substituted |

The requested Task 2 final English manuscript was searched for in the D project and canonical C worktree but not located. The user's desktop `问题二_Pre-LightGBM论文正文（用过skill写的文章）.md` was identified as Chinese, read for context, and checked against the final frozen rule. Its older illustrative resource profile is not imported into Task 4. No nonexistent English source is cited. The missing English version leaves a terminology/editorial cross-check pending, not an unresolved numerical or formula discrepancy. The user was asked for its location while writing continued.

The supplied official PDF's extracted fourth modeling requirement in ../audit/problem_extracted.txt defines the six-question coverage; the original desktop PDF remains hashed. Document instructions are source context, not authorization for additional experiments or a full submission.

## 3. Mathematical and notation audit

| Paper formula | Verified source / condition |
| --- | --- |
| (5.1) Q and historical/valid pass, violation | Final Chinese report section 1, Task3 (3.22), saved quality_harms and metrics. q_(i,m_i) is historical answer score; V_i is Task3 failure/late/extra-wait indicator, extended to fixed-window censoring (none occurs in these runs). 1-V_i exactly selects compliant completions; rates divide by all inputs. Completion is a different metric. |
| (5.2) facility carbon | Task3 (3.6): C_total=C_idle+C_active+C_net+C_ctrl; C_op=total-idle. C_active is the busy-over-idle contribution; net/control are not added twice. IT PUE once; network convention unchanged. |
| (5.3) two contrasts | rebound_audit and energy_paired: matched Growth G-Sp versus cross-demand Growth Sp-Baseline G. Values are three paired-seed means, both in gCO2e. |
| (5.4) linear reference | Final Chinese report section 4.2 and rebound_audit: solve idle+(1+r)N0*c0=C_G(N0); requires common idle, c0>0 and unchanged per-input operating coefficient. Three saved substitution residuals are zero; no DES threshold follows. |
| Inline routing disparity | Final Chinese report section 5.1: mean paired q_selected-q_L within group. Signed net difference, not positive-part loss and not a scheduling penalty. L does not dominate S pointwise. |

Task3 uses section 4 in its current English manuscript. This chapter therefore uses section 5 and formula tags (5.1)-(5.4), avoiding duplicate upstream numbering while preserving m_i, q_i, V_i, H, C_total/C_idle/C_op and all accounting units. C_Sp(N) is an explicit local abbreviation for Spatial's facility total, avoiding confusion with language-model S; it changes no model. The TeX fragment introduces no document class or packages. It reuses the existing amsmath/graphicx/longtable/booktabs/caption/etoolbox environment, namespaces labels, and uses automatic table counters/references. Equation tags follow the present manuscript numbering; renumber only if the full document structure changes.

## 4. Figure and environmental-source audit

| Item | Source, units and caption check |
| --- | --- |
| Figure 1 | figures/Figure1_Robustness.png unchanged. Four panels: selected historical Score, valid delivered pass %, completed P95 s, all-input violation %. Points=three seeds, bars=observed range, not CI. Difficulty label explicitly means Math/Code share. Source: original Results and existing plotting script. |
| Figure 2 | figures/Figure2_Carbon_Growth.png unchanged. Left kgCO2e including fixed idle; right gCO2e above idle. Within-demand G/Spatial and cross-demand growth comparisons explained separately. Original plotted means round consistently with the audited totals. |
| Fairness presentation | Table 1 replaces Figure3_Fairness.png to keep three pages and show all four service metrics. The existing third figure was checked, remains unmodified, and is not an extra paper result. Its Spatial-minus-Greedy percentage-point direction agrees with the table. |
| Environmental evidence | Table 2 condenses Task4_Environmental_Evidence.csv and audit/public_evidence_sources.json, together with the Final Chinese report section 6. No new external measurement or bibliography entry is invented. The inherited published-power proxy is transferred, not a target-model measurement; idle/shared power/PUE/arrivals/service are assumed. |

Historical grid observations and RouterBench outcomes support conditional attributional comparisons. Original forecast vintages, representative public-sector workloads, target hardware/site PUE, marginal emissions and appropriate causal counterfactuals remain missing. The small +2.496g reversal has not been validated against parameter uncertainty. Rebound is imposed demand growth, not identified efficiency-induced adoption. No production-risk coverage, demographic fairness or deployment certification is claimed.

The Markdown and TeX use relative figure paths. Compile the fragment from simple_assessment, or retain its figures/ directory when moving it into the main paper. Original figures/data are not rewritten. The preview uses the existing Task3 academic-revision preamble (10pt A4, 18mm margins, same fonts and spacing); only its Task3 header is retitled Task4. Figure 1 is placed after the opening carbon paragraph in the paginated version to avoid an empty page; the prose and figure data remain identical. The PDF has three pages including two figures, two compact tables and four displayed formulas. Typesetting logs, including earlier failed drafts, are retained under paper_writing/tmp/pdfs.

## 5. Preservation and acceptance

The pre-writing snapshot paper_writing/before.json is checked at delivery: every pre-existing TASK4 file, all D upstream files, used C final-release source/data, and the contextual desktop files retain their content hashes. Original 24 run artifacts, prior final-review outputs, all three original images and historical failure logs are unchanged. This pass has no simulation/inference/training entry point.

All numerical claims and mathematical identities used above pass the saved-data audit. The PDF's latest three pages were rendered and individually inspected: formulas, axes, legends, captions and both tables are legible, with no clipped text, missing ending or split table. That result and the final file hashes are recorded in paper_writing/delivery_validation.json. No unresolved numerical or formula issue was identified. The only missing requested source is the separately named Task2 final English prose; its frozen implementation and Chinese source are available and verified.
