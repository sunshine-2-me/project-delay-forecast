# Project Delay Forecasting

This project implements a machine learning pipeline to predict task-level delays and estimate project-level makespan using Monte-Carlo simulation.

## Overview

The solution consists of two main components:

1. **ML Predictor (Task-Level)**: A Random Forest regressor that predicts task-level delays based on task features
2. **Monte-Carlo Simulator (Project-Level)**: Simulates project makespan by propagating task durations across each project's DAG

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

To train the model and generate all predictions:

```bash
python main.py
```

This will:
1. Train the task-level delay predictor on training data
2. Generate task delay predictions for test data
3. Run Monte-Carlo simulation to estimate project makespan
4. Save outputs to `out/preds/`

### Output Files

The pipeline generates two prediction files in `out/preds/`:

1. **task_delay_preds.parquet**: Task-level delay predictions
   - Columns: `project_id`, `task_id`, `pred_task_delay_days`

2. **project_makespan_preds.parquet**: Project-level makespan predictions
   - Columns: `project_id`, `pred_makespan_mean`, `pred_makespan_p50`, `pred_makespan_p95`

### Running Components Separately

You can also run the components individually:

```bash
# Train and generate task delay predictions only
python task_predictor.py

# Run Monte-Carlo simulation (requires task_delay_preds.parquet to exist)
python monte_carlo_simulator.py
```

## Model Choices

### Task-Level Delay Predictor

**Model**: Random Forest Regressor

**Rationale**:
- Random Forests handle non-linear relationships and feature interactions well
- Robust to outliers and missing values
- Good performance on tabular data without extensive hyperparameter tuning
- Provides feature importance insights

**Hyperparameters**:
- `n_estimators=200`: Good balance between accuracy and training time
- `max_depth=15`: Prevents overfitting while capturing complex patterns
- `min_samples_split=10`, `min_samples_leaf=5`: Regularization parameters

**Features Used**:
- Planned duration and start offset
- Date features (month, day of year, day of week)
- Encoded categorical features (crew name, region)
- Text features from task labels (length, word count)

**Preprocessing**:
- Label encoding for categorical variables
- Standard scaling for numeric features
- Handling of unseen categories in test data

### Monte-Carlo Simulator

**Approach**: Monte-Carlo simulation with 1000 iterations

**Rationale**:
- Accounts for uncertainty in task duration predictions
- Properly handles DAG dependencies through topological sorting
- Provides distributional estimates (mean, median, P95) rather than point estimates

**Simulation Details**:
- Task durations are modeled as: `actual_duration = planned_duration + delay`
- Delay is sampled from a normal distribution centered on predicted delay
- Standard deviation scales with planned duration and predicted delay
- DAG constraints are enforced: tasks cannot start until all predecessors finish
- Project makespan = maximum finish time across all tasks

**Topological Sorting**:
- Uses NetworkX to build and validate DAGs
- Ensures tasks are processed in correct dependency order
- Handles isolated tasks (no dependencies) gracefully

**Duration Distribution**:
- Normal distribution for task delays
- Mean: predicted delay from ML model
- Standard deviation: `max(1.0, |predicted_delay| * 0.2 + planned_duration * 0.1)`
- Minimum duration: 1 day (actual durations cannot be negative)

## Project Structure

```
project-delay-forecast/
├── data/
│   ├── train/
│   │   ├── tasks_train.parquet
│   │   ├── projects_train.parquet
│   │   └── dependencies_train.parquet
│   └── test/
│       ├── tasks_test_inputs.parquet
│       ├── projects_test_inputs.parquet
│       └── dependencies_test.parquet
├── models/
│   └── task_delay_predictor.joblib
├── out/
│   └── preds/
│       ├── task_delay_preds.parquet
│       └── project_makespan_preds.parquet
├── task_predictor.py
├── monte_carlo_simulator.py
├── main.py
├── requirements.txt
└── README.md
```

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning models
- pyarrow: Parquet file I/O
- networkx: Graph operations for DAG handling

## Notes

- The model assumes task delays can be negative (tasks finishing early)
- The simulator enforces minimum task duration of 1 day
- DAG validation ensures no cycles in dependency graphs
- Unseen categories in test data are mapped to 'unknown' during encoding

## Future Improvements

Potential enhancements:
- Feature engineering: project-level aggregations, task complexity metrics
- Model improvements: Gradient Boosting (XGBoost/LightGBM), hyperparameter optimization
- Simulation improvements: task correlation modeling, resource constraints
- Validation: Cross-validation, holdout set evaluation
