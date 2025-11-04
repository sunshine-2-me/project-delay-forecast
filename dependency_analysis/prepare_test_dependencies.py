"""
Prepare test dependencies data similar to training data preparation.

This script:
1. Reshapes test dependencies to wide format (one row per task with predecessor columns)
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


def create_test_dependencies_with_counts(deps_file, tasks_file, output_dir):
    """
    Create wide-format test dependencies with predecessor counts and delays.
    
    Args:
        deps_file: Path to test dependencies parquet file
        tasks_file: Path to test tasks parquet file (or task delay predictions)
        output_dir: Directory to save output files
    """
    print("="*60)
    print("Preparing Test Dependencies")
    print("="*60)
    
    print(f"\nLoading dependencies from {deps_file}...")
    deps = pd.read_parquet(deps_file)
    print(f"Total dependency rows: {len(deps):,}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Reshape to wide format
    print("\nReshaping dependencies to wide format...")
    deps_wide = reshape_dependencies_to_wide_format(deps)
    
    # Add predecessor count
    pred_id_cols = [col for col in deps_wide.columns if col.startswith('pred_task_id_')]
    deps_wide['num_predecessors'] = deps_wide[pred_id_cols].notna().sum(axis=1)
    
    # Load task delays (from tasks file or predictions)
    print(f"\nLoading task delays from {tasks_file}...")
    tasks = pd.read_parquet(tasks_file)
    print(f"Total tasks: {len(tasks):,}")
    
    # Check if we have delay information (could be 'task_delay_days' or 'pred_task_delay_days')
    delay_col = None
    if 'task_delay_days' in tasks.columns:
        delay_col = 'task_delay_days'
    elif 'pred_task_delay_days' in tasks.columns:
        delay_col = 'pred_task_delay_days'
    
    # If no delay column, try to calculate from actual vs planned duration
    if delay_col is None:
        if 'actual_duration_days' in tasks.columns and 'planned_duration_days' in tasks.columns:
            print("Calculating delays from actual_duration_days - planned_duration_days...")
            tasks['task_delay_days'] = tasks['actual_duration_days'] - tasks['planned_duration_days']
            delay_col = 'task_delay_days'
            print(f"  Mean delay: {tasks['task_delay_days'].mean():.3f}")
            print(f"  Non-null delays: {tasks['task_delay_days'].notna().sum():,}")
        else:
            print("Warning: No delay column found and cannot calculate from durations. Delay columns will be NaN.")
            # Create a task_delays DataFrame with NaN delays
            task_delays = tasks[['project_id', 'task_id']].copy()
            task_delays['task_delay_days'] = np.nan
    
    if delay_col is not None:
        # Extract delay information
        task_delays = tasks[['project_id', 'task_id', delay_col]].copy()
        if delay_col != 'task_delay_days':
            task_delays.rename(columns={delay_col: 'task_delay_days'}, inplace=True)
        if delay_col == 'task_delay_days' and delay_col not in tasks.columns:
            # Already calculated above
            task_delays = tasks[['project_id', 'task_id', 'task_delay_days']].copy()
    
    # Extract planned duration information
    if 'planned_duration_days' in tasks.columns:
        task_planned = tasks[['project_id', 'task_id', 'planned_duration_days']].copy()
        print(f"Tasks with planned duration: {len(task_planned):,}")
    else:
        print("Warning: planned_duration_days not found in tasks file. Planned duration columns will be NaN.")
        task_planned = tasks[['project_id', 'task_id']].copy()
        task_planned['planned_duration_days'] = np.nan
    
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
    for pred_col in pred_id_cols:
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
    
    # Reorder columns: base columns, then interleaved pred_id, pred_delay, and pred_planned
    base_cols = ['project_id', 'succ_task_id', 'succ_task_delay_days', 'succ_task_planned_days', 'num_predecessors']
    pred_delay_cols = sorted([col for col in deps_wide.columns if col.startswith('pred_task_delay_days_')],
                              key=lambda x: int(x.split('_')[-1]))
    pred_planned_cols = sorted([col for col in deps_wide.columns if col.startswith('pred_task_planned_days_')],
                                key=lambda x: int(x.split('_')[-1]))
    
    # Interleave pred_task_id, pred_task_delay_days, and pred_task_planned_days
    interleaved_pred_cols = []
    for i in range(len(pred_id_cols)):
        interleaved_pred_cols.append(pred_id_cols[i])
        if i < len(pred_delay_cols):
            interleaved_pred_cols.append(pred_delay_cols[i])
        if i < len(pred_planned_cols):
            interleaved_pred_cols.append(pred_planned_cols[i])
    
    final_cols = base_cols + interleaved_pred_cols
    deps_wide = deps_wide[final_cols]
    
    # Sort by project_id and succ_task_id
    deps_wide = deps_wide.sort_values(['project_id', 'succ_task_id']).reset_index(drop=True)
    
    # Save dependencies with counts
    deps_output_file = Path(output_dir) / 'test_dependencies.parquet'
    deps_wide.to_parquet(deps_output_file, index=False)
    
    print(f"\nDependencies saved to: {deps_output_file}")
    print(f"  Total rows: {len(deps_wide):,}")
    print(f"  Columns: {deps_wide.columns.tolist()}")
    
    # Show distribution of predecessor counts
    print(f"\nDistribution of predecessor counts:")
    pred_dist = deps_wide['num_predecessors'].value_counts().sort_index()
    print(pred_dist)
    
    # Show sample of dependencies
    print(f"\nSample dependencies (first 5 rows, showing key columns):")
    sample_cols = ['project_id', 'succ_task_id', 'succ_task_delay_days', 'succ_task_planned_days', 'num_predecessors']
    pred_cols_to_show = [col for col in deps_wide.columns if 'pred_task' in col][:9]  # Show first 2 pred sets (id, delay, planned)
    print(deps_wide[sample_cols + pred_cols_to_show].head(5).to_string(index=False))
    
    # Show delay statistics
    if 'succ_task_delay_days' in deps_wide.columns:
        print(f"\nDelay statistics:")
        print(f"  Successor task delays - mean: {deps_wide['succ_task_delay_days'].mean():.3f}, "
              f"non-null: {deps_wide['succ_task_delay_days'].notna().sum():,}")
        if 'pred_task_delay_days_1' in deps_wide.columns:
            print(f"  Predecessor 1 delays - mean: {deps_wide['pred_task_delay_days_1'].mean():.3f}, "
                  f"non-null: {deps_wide['pred_task_delay_days_1'].notna().sum():,}")
    
    # Show planned duration statistics
    if 'succ_task_planned_days' in deps_wide.columns:
        print(f"\nPlanned duration statistics:")
        print(f"  Successor task planned - mean: {deps_wide['succ_task_planned_days'].mean():.3f}, "
              f"non-null: {deps_wide['succ_task_planned_days'].notna().sum():,}")
        if 'pred_task_planned_days_1' in deps_wide.columns:
            print(f"  Predecessor 1 planned - mean: {deps_wide['pred_task_planned_days_1'].mean():.3f}, "
                  f"non-null: {deps_wide['pred_task_planned_days_1'].notna().sum():,}")
    
    return deps_wide


def main():
    """Main function."""
    print("="*60)
    print("Prepare Test Dependencies")
    print("="*60)
    
    # Configuration
    deps_test_file = 'data/test/dependencies_test.parquet'
    tasks_test_file = 'data/test/tasks_test_inputs.parquet'
    output_dir = 'out'
    
    # Check if input files exist
    if not Path(deps_test_file).exists():
        print(f"\nError: Dependencies file not found at {deps_test_file}")
        return
    
    if not Path(tasks_test_file).exists():
        print(f"\nError: Tasks file not found at {tasks_test_file}")
        return
    
    # Create dependencies
    deps_wide = create_test_dependencies_with_counts(
        deps_test_file, tasks_test_file, output_dir
    )
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Output file saved to: {output_dir}/")
    print(f"  test_dependencies.parquet")
    print(f"    - {len(deps_wide):,} tasks with dependencies")
    
    # File size info
    deps_file = Path(output_dir) / 'test_dependencies.parquet'
    if deps_file.exists():
        size_mb = deps_file.stat().st_size / (1024 * 1024)
        print(f"    - File size: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
