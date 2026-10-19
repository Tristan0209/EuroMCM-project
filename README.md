# EuroMCM Project

This repository contains the code, data, models, and experimental results supporting our EuroMCM submission. All submission materials are collected in **[Our work](./Our%20work/)**.

The work covers four connected parts: modeling the quality and resource costs of language model inference, selecting models through adaptive routing, scheduling requests across regions, and evaluating the resulting system under different operating conditions.

## Folder Overview

| Folder | Main contents |
| --- | --- |
| `Our work/Work 1/` | Quality prediction, response assessment, cost analysis, latency simulation, and environmental impact models |
| `Our work/task2/` | Adaptive routing methods, calibration, trained models, and routing evaluation results |
| `Our work/TASK 3/` | Scheduling inputs, optimization models, discrete-event simulation, and scheduling results |
| `Our work/TASK 4/` | Scenario experiments, comparisons between scheduling strategies, and detailed execution records |
| `Our work/AI_datasets/` | Datasets and reference inputs used in the analysis |

## Task 1: Quality, Cost, and Environmental Impact

`Work 1/` contains the models used to describe the trade-offs between answer quality and the resources required to serve a request.

The quality analysis includes several prediction methods, including LightGBM, XGBoost, and item response theory models. The response assessment modules examine generated answers using task-specific features and diagnostics.

The resource models cover inference cost, response time, energy consumption, and carbon emissions. The folder also contains sensitivity analyses for power assumptions and the data used to produce the corresponding figures.

The main files and subfolders include:

- `task1_quality_pre/`: quality prediction models based on information available before inference.
- `task_1_quality_post/`: response assessment models, saved assessment models, and supporting results.
- `task1_response time, and environ mental impact/`: latency and environmental impact models, processed trace data, simulations, and validation records.
- `task1_cost.py` and `cost_dgx_engine.py`: cost and power accounting.
- `chapter3_figures/`: plotting scripts, figure data, and exported figures.
- `outputs/`: cost summaries, power sensitivity tables, validation results, and figures.

Saved results include cost breakdowns, latency summaries, energy and emissions estimates, and comparisons across workload and resource configurations.

## Task 2: Adaptive Model Routing

`task2/` contains the routing framework that selects between a smaller and a larger language model. It combines predicted quality with cost estimates and calibrated fallback rules.

This folder includes the training and evaluation code, the input data used for routing, saved models, and the results of comparisons between routing methods.

Its main components are:

- `inputs/`: paired requests and outcomes, data partitions, and frozen input features.
- `support/`: feature processing code and supporting comparison models.
- `runs/experiment/`: the retained experiment, including trained models, selected parameters, calibration records, predictions, and routing decisions.
- `outputs/`: additional analysis tables and figures.

The saved results describe model choices, quality and cost trade-offs, fallback behavior, and performance across task categories. Calibration and provenance records are included alongside the results.

## Task 3: Scheduling and Carbon-Aware Optimization

`TASK 3/` contains the data preparation and scheduling models used to allocate requests across regions while accounting for service constraints and carbon intensity.

The folder is divided into two main parts:

- `data_preparation/`: preparation and validation of request data, regional carbon intensity inputs, server configurations, network parameters, and availability scenarios.
- `task3_model/`: optimization models, discrete-event simulation, energy accounting, tests, and saved experiments.

The scheduling analysis considers request timing, queueing, regional resource availability, and the carbon intensity of electricity. Results include comparisons between scheduling strategies and sensitivity analyses for demand, delay limits, forecast conditions, and resource availability.

The main result locations are:

- `task3_model/results/data/`: the inputs used by the scheduling models.
- `task3_model/results/`: aggregate comparisons, experiment tables, and validation records.
- `task3_model/results/runs/`: detailed records from the base scheduling experiments.
- `task3_model/results/robustness_v2/final_release/`: the final Spatial MILP implementation and its comparison tables.

`FINAL_COMPARISON.csv` in the final release folder contains the saved comparison results for that version. The base scheduling experiments and the final Spatial MILP release are retained separately.

## Task 4: Scenario Evaluation

`TASK 4/` examines how the routing and scheduling system behaves when the workload or prediction conditions change.

The final study is stored in `simple_assessment/`. It covers four scenarios:

| Scenario | Change examined |
| --- | --- |
| Baseline | Reference workload and prediction settings |
| Difficult | A larger share of code and mathematics requests |
| Prediction Error | Additional error in the quality predictions |
| Demand Growth | A 50% increase in the number of requests |

Each scenario uses three fixed seeds and compares Greedy scheduling with Spatial MILP, giving 24 strategy executions in total.

The main result files include:

- `Task4_Simple_Results.csv`: overall scenario and strategy results.
- `Task4_Simple_Paired_Differences.csv`: paired comparisons between the two strategies.
- `Task4_Simple_Fairness.csv`: results across request groups.
- `Task4_Simple_Rebound.csv`: demand growth and rebound analysis.
- `Task4_Simple_Server_Load.csv`: server workload summaries.
- `Task4_Environmental_Evidence.csv`: supporting environmental accounting results.
- `runs/`: saved worlds, execution records, metrics, and event-level outputs.

The remaining modules in `task4_core/` support scenario generation, routing, scheduling, resource accounting, and result validation. Plotting code and exported figures are also included.

## Data and Saved Records

`AI_datasets/` contains supporting inputs, including RouterBench data and reference material for energy and electricity carbon intensity. Additional processed inputs are stored beside the models that use them.

The full original Azure trace files are not included; processed trace inputs are retained in the Task 1 folder. The complete upstream execution archive for the final Spatial MILP release is also not duplicated in this repository.

Saved experimental records preserve their original parameters, seeds, source information, and file hashes. Some records contain paths from the computing environment in which the experiments were performed.

The submission also includes dependency files, automated checks, and a file integrity manifest. These support inspection of the code and saved results. External datasets remain subject to their respective licenses.
