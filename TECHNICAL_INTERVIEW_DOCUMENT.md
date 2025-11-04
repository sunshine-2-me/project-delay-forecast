# Technical Interview: Project Delay Forecasting Solution
## Oral Presentation Format

---

## Introduction

Hi, I'm excited to walk you through the solution I've developed for predicting task-level delays in project management. Let me break down the approach in a way that's easy to follow.

The core idea is pretty straightforward: when a task has predecessors that get delayed, those delays tend to cascade down to dependent tasks. So, if I know how delayed the predecessor tasks are, I can predict how much the successor task will be delayed.

---

## Part 1: Understanding the Data Structure

### The Challenge

First, let me explain the data challenge we're dealing with. We have tasks that depend on other tasks, forming what's called a Directed Acyclic Graph, or DAG. Think of it like a project plan where Task B can't start until Task A finishes.

The tricky part is that tasks can have different numbers of predecessors. Some tasks might depend on just one predecessor, others on two, three, or even more. This variable structure makes it hard to feed directly into a machine learning model, which typically expects a fixed number of features.

### The Input Data Format

Our raw data comes in what's called "long format" - basically, one row per dependency relationship:

```
project_id | pred_task_id | succ_task_id
1          | 4            | 7
1          | 5            | 7
1          | 6            | 7
```

So here, task 7 depends on three different tasks: 4, 5, and 6. But we also have task information separately, which includes things like how many days each task was delayed, and what the planned duration was.

---

## Part 2: Preparing the Training Dataset

### Step 1: Reshaping from Long to Wide Format

The first thing we need to do is transform this long format into what I call "wide format" - where each row represents one successor task, and all its predecessors are spread out into columns.

So from the example above, we'd get:
```
project_id | succ_task_id | pred_task_id_1 | pred_task_id_2 | pred_task_id_3
1          | 7            | 4              | 5              | 6
```

The script `prepare_train_dependencies.py` does this transformation. It:
- Groups all dependencies by successor task
- Ranks the predecessors (we sort them by task ID to keep it consistent)
- Pivots the data so each predecessor gets its own column: `pred_task_id_1`, `pred_task_id_2`, `pred_task_id_3`, and so on

### Step 2: Adding Delay and Duration Information

Now, having just the task IDs isn't enough - we need the actual delay values and planned durations. So the script then merges in information from the tasks dataset.

For each predecessor, we add:
- `pred_task_delay_days_1`, `pred_task_delay_days_2`, etc. - how many days each predecessor was delayed
- `pred_task_planned_days_1`, `pred_task_planned_days_2`, etc. - the planned duration of each predecessor

And for the successor task itself, we add:
- `succ_task_delay_days` - this is our target variable, what we want to predict
- `succ_task_planned_days` - the planned duration of the successor task

### Step 3: Handling Variable Predecessor Counts

Here's where it gets interesting. What happens if Task A has 3 predecessors but Task B only has 1? Task B will have `NaN` values for `pred_task_id_2` and `pred_task_id_3`.

Our solution: we fill those `NaN` values with zero. The logic is simple - if there's no predecessor in that slot, it shouldn't contribute to the delay. The model will learn to ignore these zero-filled features.

The final training dataset looks like this:
```
project_id | succ_task_id | succ_task_delay_days | succ_task_planned_days | 
           | pred_task_id_1 | pred_task_delay_days_1 | pred_task_planned_days_1 |
           | pred_task_id_2 | pred_task_delay_days_2 | pred_task_planned_days_2 |
           | ... | num_predecessors
```

This gets saved as `out/train_dependencies.parquet`.

---

## Part 3: Training the Model

### Feature Selection

Now that we have our training data in the right format, let's talk about what features the model actually uses.

The model uses three types of features:
1. **Predecessor delays** - `pred_task_delay_days_1`, `pred_task_delay_days_2`, etc. These are the primary signals. If a predecessor is delayed, the successor is likely to be delayed too.

2. **Successor planned duration** - `succ_task_planned_days`. Longer tasks might have different delay patterns - maybe they have more buffer time built in.

3. **Predecessor planned durations** - `pred_task_planned_days_1`, etc. This captures interactions - like if a short task depends on a very long task, the delay propagation might be different.

The target variable we're predicting is `succ_task_delay_days` - how many days the successor task was actually delayed compared to its plan.

### Model Choice: Why Gradient Boosting?

I chose LightGBM, which is a gradient boosting model. Here's why:

First, it handles non-linear relationships really well. Delay propagation isn't necessarily linear - maybe if you have two predecessors both delayed by 5 days, the impact isn't just 10 days. There might be interactions, thresholds, or other complex patterns.

Second, it's great at handling this kind of tabular data where we have a mix of features with different relationships.

Third, it provides feature importance, which helps us understand which predecessors matter most.

And finally, it's robust - it can handle missing values, outliers, and doesn't require a lot of data preprocessing like scaling.

### The Training Process

The training script `train_task_delay_model.py` does the following:

1. **Loads the prepared training data** from `out/train_dependencies.parquet`

2. **Extracts features and target**: It identifies all the predecessor delay columns, planned duration columns, and separates them from the target variable.

3. **Handles missing values**: Fills any remaining `NaN` values with 0 (for tasks with fewer predecessors).

4. **Splits into train and validation**: We use an 80/20 split - 80% for training, 20% for validation to check how well the model generalizes.

5. **Trains the model**: The model uses these hyperparameters:
   - 500 boosting rounds (but with early stopping, so it might stop earlier)
   - Maximum depth of 7 - this prevents overfitting while still capturing complex patterns
   - Learning rate of 0.05 - conservative, but with early stopping it works well
   - Early stopping after 50 rounds with no improvement - this prevents overfitting

6. **Evaluates performance**: We calculate three metrics:
   - **MAE (Mean Absolute Error)**: Average error in days - very interpretable
   - **RMSE (Root Mean Squared Error)**: Penalizes large errors more
   - **R² Score**: How much variance the model explains

7. **Saves the model**: The trained model, along with metadata like feature column names and model type, gets saved as `out/models/delay_predictor_lightgbm.joblib`.

Why save the feature column names? That's important for prediction - we need to make sure the test data has features in the exact same order as training.

---

## Part 4: Preparing Test Data and Making Predictions

### Preparing Test Dependencies

The test data preparation follows almost exactly the same process as training data, but there's one key difference.

The script `prepare_test_dependencies.py` does the same long-to-wide transformation, but when it comes to delays, test data might not have actual delays yet - that's what we're trying to predict! However, if the test data has `actual_duration_days` and `planned_duration_days`, we can calculate the delay ourselves: `task_delay_days = actual_duration_days - planned_duration_days`.

This gives us the predecessor delays we need for making predictions. The output is saved as `out/test_dependencies.parquet` with the same structure as the training data.

### Making Predictions

Now comes the prediction step with `predict_task_delays.py`:

1. **Load the trained model**: We deserialize the model file, which includes the model itself and the feature column names.

2. **Load test dependencies**: Read in the prepared test data.

3. **Extract and align features**: This is crucial - we need to make sure:
   - We use the same features in the same order as training
   - If test data has fewer predecessor columns than training, we add zeros
   - If test data has more (unlikely but possible), we handle it
   - All `NaN` values are filled with 0

4. **Generate predictions**: Pass the feature matrix through the model to get predicted delays for each successor task.

5. **Save results**: Add the predictions as a new column `pred_succ_task_delay_days` right next to the actual `succ_task_delay_days` (if it exists), and save everything as `out/test_dependencies_predictions.parquet`.

### Why Feature Alignment Matters

This is worth emphasizing - feature alignment is critical. The model learned on a specific set of features in a specific order. If we change that order or add/remove features, the predictions will be wrong. That's why we save the feature column names with the model and carefully ensure test data matches.

---

## Part 5: Future Enhancement - Monte Carlo Simulation

Now, let me explain how we can extend this solution to estimate project-level makespan using Monte Carlo simulation. This is where the solution becomes really powerful.

### The Motivation

Right now, we're predicting delays for individual tasks. But in project management, you often want to know: "If these tasks have delays, what does that mean for the overall project completion time?"

That's where Monte Carlo simulation comes in. Instead of just using point predictions (like "this task will be delayed by 3 days"), we can model the uncertainty and simulate thousands of possible project outcomes.

### How It Would Work

Here's my approach for implementing Monte Carlo simulation:

#### Step 1: Prepare Task Duration Distributions

For each task, instead of a single delay prediction, we'd create a distribution. We could model it as:
- **Mean**: The predicted delay from our model
- **Standard deviation**: Based on the model's prediction error or uncertainty

For example, if the model predicts a 5-day delay with an RMSE of 2 days, we might model it as a normal distribution with mean=5 and std=2.

#### Step 2: Topological Sort of Tasks

Since tasks form a DAG, we need to process them in the right order - a task can't start until all its predecessors finish. We'd use topological sorting (using NetworkX library) to get the correct sequence.

#### Step 3: The Simulation Loop

For each simulation iteration (say, 1000 iterations):

1. **Sample task durations**: For each task, sample a delay from its distribution. The actual duration would be `planned_duration + sampled_delay`.

2. **Calculate start times**: For each task, find the maximum finish time of all its predecessors. That becomes the task's start time.

3. **Calculate finish times**: Start time + actual duration = finish time.

4. **Calculate project makespan**: The maximum finish time across all tasks is the project completion time for this iteration.

5. **Store the result**: Save this simulated makespan.

After 1000 iterations, we'd have 1000 different possible project completion times.

#### Step 4: Aggregate Results

From these 1000 simulations, we can calculate:
- **Mean makespan**: Average completion time across all simulations
- **Median (P50)**: The 50th percentile - half the time we finish before this, half after
- **P95 makespan**: The 95th percentile - 95% of simulations finish before this time (useful for risk management)

### Integration with Our Current Solution

The beautiful part is how this integrates with what we've already built:

1. **Use our delay predictions**: For each task in a project, we'd use our trained model to predict delays for tasks with predecessors.

2. **Handle initial tasks differently**: Tasks with no predecessors - we'd need a separate model or baseline prediction (maybe average historical delays, or zero if we have no data).

3. **Propagate through the DAG**: The simulation would use the topological order to ensure delays cascade correctly.

4. **Account for uncertainty**: Instead of just using the predicted delay as-is, we'd sample from a distribution centered on that prediction.

### Why This Adds Value

This approach gives project managers much richer information:
- **Not just "the project will take 100 days"**, but "there's a 50% chance it takes less than 100 days, and a 95% chance it takes less than 115 days"
- **Risk assessment**: The difference between mean and P95 tells you how much uncertainty there is
- **What-if scenarios**: You could simulate "what if we add more resources to these tasks?" by modifying the delay distributions

### Implementation Considerations

To implement this, we'd need to:

1. **Extend the prediction script**: Generate delay distributions, not just point estimates. This might mean using quantile regression or modeling prediction intervals.

2. **Build a simulation engine**: A new module that:
   - Loads project DAG structure
   - Performs topological sorting
   - Runs the simulation loop
   - Aggregates and reports results

3. **Handle edge cases**: Tasks with no predecessors, isolated tasks, circular dependencies (though DAGs shouldn't have these).

4. **Optimize performance**: 1000 simulations × hundreds of tasks could be slow, so we might need parallel processing or optimized sampling.

This Monte Carlo extension transforms our solution from task-level delay prediction to full project risk assessment, which is much more actionable for project managers.

---

## Summary

So to summarize the entire solution:

**Data Preparation**: We transform long-format dependencies into wide-format feature matrices, enriching them with delay and duration information. This handles variable numbers of predecessors by using a fixed-width format with zero-padding.

**Training**: We use gradient boosting (LightGBM) to learn how predecessor delays and durations influence successor delays. The model captures non-linear relationships and interactions.

**Prediction**: We prepare test data in the same format, ensure feature alignment, and generate delay predictions using the trained model.

**Future Enhancement**: Monte Carlo simulation would allow us to propagate these task-level predictions through the entire project DAG, giving probabilistic makespan estimates rather than just point predictions.

The solution is modular, scalable, and ready for production use. And with the Monte Carlo extension, it becomes a comprehensive project risk assessment tool.

Thank you for listening! I'm happy to answer any questions about the approach, implementation details, or potential improvements.
