"""
Split dependencies data by number of predecessors.

This script creates a wide format file with all dependencies,
including delay information for successor and predecessor tasks.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def reshape_dependencies_to_wide_format(deps):
    """
    Reshape dependencies from long to wide format.
    
    Converts from:
        project_id  pred_task_id  succ_task_id
        1           4             7
        1           5             7
        1           6             7
    
    To:
        project_id  succ_task_id  pred_task_id_1  pred_task_id_2  pred_task_id_3
        1           7             4               5               6
    """
    # Sort by pred_task_id to ensure consistent ordering
    deps_sorted = deps.sort_values(['project_id', 'succ_task_id', 'pred_task_id']).copy()
    
    # Add a rank for each predecessor
    deps_sorted['pred_rank'] = deps_sorted.groupby(['project_id', 'succ_task_id']).cumcount() + 1
    
    # Pivot to wide format
    deps_wide = deps_sorted.pivot_table(
        index=['project_id', 'succ_task_id'],
        columns='pred_rank',
        values='pred_task_id',
        aggfunc='first'
    ).reset_index()
    
    # Rename columns
    deps_wide.columns = ['project_id', 'succ_task_id'] + [f'pred_task_id_{i}' for i in range(1, len(deps_wide.columns) - 1)]
    
    return deps_wide


def create_dependencies_with_delays(deps_file, tasks_file, output_dir):
    """
    Create a file with all dependencies in wide format, including delay and planned duration information.
    
    Args:
        deps_file: Path to dependencies parquet file
        tasks_file: Path to tasks parquet file (to get task_delay_days and planned_duration_days)
        output_dir: Directory to save output file
    """
    print(f"Loading dependencies from {deps_file}...")
    deps = pd.read_parquet(deps_file)
    print(f"Total dependency rows: {len(deps):,}")
    
    print(f"\nLoading tasks from {tasks_file}...")
    tasks = pd.read_parquet(tasks_file)
    print(f"Total tasks: {len(tasks):,}")
    
    # Extract delay information (project_id, task_id, task_delay_days)
    task_delays = tasks[['project_id', 'task_id', 'task_delay_days']].copy()
    print(f"Tasks with delay information: {len(task_delays):,}")
    
    # Extract planned duration information
    if 'planned_duration_days' in tasks.columns:
        task_planned = tasks[['project_id', 'task_id', 'planned_duration_days']].copy()
        print(f"Tasks with planned duration: {len(task_planned):,}")
    else:
        print("Warning: planned_duration_days not found in tasks file. Planned duration columns will be NaN.")
        task_planned = tasks[['project_id', 'task_id']].copy()
        task_planned['planned_duration_days'] = np.nan
    
    # Count predecessors per task
    print("\nCounting predecessors per task...")
    pred_counts = deps.groupby(['project_id', 'succ_task_id']).size().reset_index(name='num_predecessors')
    
    print("\nDistribution of predecessor counts:")
    dist = pred_counts['num_predecessors'].value_counts().sort_index()
    print(dist)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Reshape all dependencies to wide format
    print("\n" + "="*60)
    print("Reshaping to wide format...")
    print("="*60)
    
    deps_wide = reshape_dependencies_to_wide_format(deps)
    
    # Add successor task delay
    print("\nAdding successor task delays...")
    deps_wide = deps_wide.merge(
        task_delays.rename(columns={'task_id': 'succ_task_id', 'task_delay_days': 'succ_task_delay_days'}),
        on=['project_id', 'succ_task_id'],
        how='left'
    )
    
    # Add successor task planned duration
    print("Adding successor task planned durations...")
    deps_wide = deps_wide.merge(
        task_planned.rename(columns={'task_id': 'succ_task_id', 'planned_duration_days': 'succ_task_planned_days'}),
        on=['project_id', 'succ_task_id'],
        how='left'
    )
    
    # Add predecessor delays and planned durations
    print("Adding predecessor task delays and planned durations...")
    pred_cols = [col for col in deps_wide.columns if col.startswith('pred_task_id_')]
    
    for pred_col in pred_cols:
        # Get the rank number (e.g., 1, 2, 3)
        rank = pred_col.split('_')[-1]
        delay_col = f'pred_task_delay_days_{rank}'
        planned_col = f'pred_task_planned_days_{rank}'
        
        # Merge delay for this predecessor
        deps_wide = deps_wide.merge(
            task_delays.rename(columns={'task_id': pred_col, 'task_delay_days': delay_col}),
            on=['project_id', pred_col],
            how='left'
        )
        
        # Merge planned duration for this predecessor
        deps_wide = deps_wide.merge(
            task_planned.rename(columns={'task_id': pred_col, 'planned_duration_days': planned_col}),
            on=['project_id', pred_col],
            how='left'
        )
    
    # Add predecessor count
    deps_wide = deps_wide.merge(
        pred_counts,
        on=['project_id', 'succ_task_id'],
        how='left'
    )
    
    # Reorder columns: project_id, succ_task_id, succ_task_delay_days, succ_task_planned_days, num_predecessors,
    # then pred_task_id_1, pred_task_delay_days_1, pred_task_planned_days_1, pred_task_id_2, pred_task_delay_days_2, pred_task_planned_days_2, ...
    base_cols = ['project_id', 'succ_task_id', 'succ_task_delay_days', 'succ_task_planned_days', 'num_predecessors']
    
    # Get pred columns in order
    pred_id_cols = sorted([col for col in deps_wide.columns if col.startswith('pred_task_id_')], 
                          key=lambda x: int(x.split('_')[-1]))
    pred_delay_cols = sorted([col for col in deps_wide.columns if col.startswith('pred_task_delay_days_')],
                             key=lambda x: int(x.split('_')[-1]))
    pred_planned_cols = sorted([col for col in deps_wide.columns if col.startswith('pred_task_planned_days_')],
                               key=lambda x: int(x.split('_')[-1]))
    
    # Interleave: pred_task_id_1, pred_task_delay_days_1, pred_task_planned_days_1, pred_task_id_2, ...
    interleaved_pred_cols = []
    for i in range(len(pred_id_cols)):
        interleaved_pred_cols.append(pred_id_cols[i])
        if i < len(pred_delay_cols):
            interleaved_pred_cols.append(pred_delay_cols[i])
        if i < len(pred_planned_cols):
            interleaved_pred_cols.append(pred_planned_cols[i])
    
    # Final column order
    final_cols = base_cols + interleaved_pred_cols
    deps_wide = deps_wide[final_cols]
    
    # Sort by project_id and succ_task_id
    deps_wide = deps_wide.sort_values(['project_id', 'succ_task_id']).reset_index(drop=True)
    
    # Save
    summary_file = Path(output_dir) / 'train_dependencies.parquet'
    deps_wide.to_parquet(summary_file, index=False)
    
    print(f"\nSaved file: {summary_file}")
    print(f"  Total rows: {len(deps_wide):,}")
    print(f"  Columns: {deps_wide.columns.tolist()}")
    print(f"  Sorted by: project_id, succ_task_id")
    
    # Show sample
    print(f"\nSample rows (first 5, showing key columns):")
    sample_cols = ['project_id', 'succ_task_id', 'succ_task_delay_days', 'succ_task_planned_days', 'num_predecessors']
    pred_cols_sample = [col for col in deps_wide.columns if 'pred_task' in col][:6]  # First 2 predecessors worth
    print(deps_wide[sample_cols + pred_cols_sample].head(5).to_string(index=False))
    
    # Show statistics on delays and planned durations
    print(f"\nDelay statistics:")
    print(f"  Successor task delays - mean: {deps_wide['succ_task_delay_days'].mean():.2f}, std: {deps_wide['succ_task_delay_days'].std():.2f}")
    if 'pred_task_delay_days_1' in deps_wide.columns:
        print(f"  Predecessor 1 delays - mean: {deps_wide['pred_task_delay_days_1'].mean():.2f}, std: {deps_wide['pred_task_delay_days_1'].std():.2f}")
    
    print(f"\nPlanned duration statistics:")
    if 'succ_task_planned_days' in deps_wide.columns:
        print(f"  Successor task planned - mean: {deps_wide['succ_task_planned_days'].mean():.2f}, std: {deps_wide['succ_task_planned_days'].std():.2f}")
    if 'pred_task_planned_days_1' in deps_wide.columns:
        print(f"  Predecessor 1 planned - mean: {deps_wide['pred_task_planned_days_1'].mean():.2f}, std: {deps_wide['pred_task_planned_days_1'].std():.2f}")
    
    return deps_wide


def main():
    """Main function."""
    deps_file = 'data/train/dependencies_train.parquet'
    tasks_file = 'data/train/tasks_train.parquet'
    output_dir = 'out'
    
    print("="*60)
    print("Creating Dependencies with Predecessor Counts and Delays")
    print("(Wide format: one row per task, sorted by project_id and succ_task_id)")
    print("="*60)
    
    deps_wide = create_dependencies_with_delays(deps_file, tasks_file, output_dir)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Output file: {output_dir}/train_dependencies.parquet")
    file_path = Path(output_dir) / 'train_dependencies.parquet'
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
