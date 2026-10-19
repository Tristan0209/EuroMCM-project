# EuroMCM-project

# EuroMCM Project

Code and experimental results for our EuroMCM submission.

All submission files are in [Our work](./Our%20work/).

## Contents

| Folder | Description |
| --- | --- |
| `Work 1` | Quality, cost, latency, energy, and carbon emissions models |
| `task2` | Adaptive routing and model evaluation |
| `TASK 3` | Scheduling models and simulation |
| `TASK 4` | Scenario experiments |
| `AI_datasets` | Input data |

## Setup

Run the following commands from the repository root:

```bash
cd "Our work"
python -m pip install -r requirements.txt
```

The code was checked with Python 3.9.13. Saved models may require compatible versions of the training libraries.

## Running the Code

Check the submitted files:

```bash
python -B validate_submission.py
```

Run a routing example:

```bash
python -B task2/predict.py --benchmark hellaswag --prompt "Choose the best continuation."
```

Run the Task 2 tests:

```bash
python -B task2/test_complete.py
```

Run the Task 4 experiments:

```bash
python -B "TASK 4/simple_assessment/run_simple.py" dry-run
python -B "TASK 4/simple_assessment/run_simple.py" run
python -B "TASK 4/simple_assessment/run_simple.py" validate
```

## Results

- Task 1: `Work 1/outputs/` and the output folders of its submodels
- Task 2: `task2/runs/experiment/`
- Task 3: `TASK 3/task3_model/results/`
- Task 4: `TASK 4/simple_assessment/Task4_Simple_Results.csv`

Detailed Task 4 records are saved in `TASK 4/simple_assessment/runs/`. New runs use separate output folders.

## Data

Processed Azure traces are included. To repeat the preprocessing from the original traces, place these files in `AI_datasets/Azure/`:

- `AzureLLMInferenceTrace_code_1week.csv`
- `AzureLLMInferenceTrace_conv_1week.csv`

Some saved records contain paths from the original experiment environment. External datasets are subject to their original licenses.
