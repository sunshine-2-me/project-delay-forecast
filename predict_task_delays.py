"""
Predict successor task delays for test data using the trained dependency-based model.

This script:
1. Loads the test dependencies with predecessor delays
2. Uses the trained DelayPredictor model to predict successor delays
3. Adds predictions to the dependencies table and saves the result
"""

import pandas as pd
import numpy as np
from pathlib import Path

from train_task_delay_model import DelayPredictor


def main():
    """Main prediction function."""
    print("="*60)
    print("Predicting Successor Task Delays")
    print("="*60)
    
    # Configuration
    input_file = 'out/test_dependencies.parquet'
    model_path = 'out/models/delay_predictor_lightgbm.joblib'  # Change if using xgboost
    output_file = 'out/test_dependencies_predictions.parquet'
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"\nError: Input file not found at {input_file}")
        print("Please run dependency_analysis/prepare_test_dependencies.py first to create the dependencies file.")
        return
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\nError: Trained model not found at {model_path}")
        print("Please train the model first by running: python train_task_delay_model.py")
        return
    
    # Load the trained model
    print(f"\nLoading trained model from {model_path}...")
    predictor = DelayPredictor()
    predictor.load(model_path)
    print(f"Model type: {predictor.model_type}")
    print(f"Features: {len(predictor.feature_columns)} ({', '.join(predictor.feature_columns)})")
    
    # Load test dependencies with delays
    print(f"\nLoading test dependencies from {input_file}...")
    deps_df = pd.read_parquet(input_file)
    print(f"Loaded {len(deps_df):,} rows")
    print(f"Columns: {deps_df.columns.tolist()}")
    
    # Prepare features for prediction
    print("\n" + "="*60)
    print("Preparing Features for Prediction")
    print("="*60)
    
    # Get all predecessor delay columns
    pred_delay_cols = [col for col in deps_df.columns if col.startswith('pred_task_delay_days_')]
    pred_delay_cols = sorted(pred_delay_cols, key=lambda x: int(x.split('_')[-1]))
    
    # Get successor planned duration
    succ_planned_cols = []
    if 'succ_task_planned_days' in deps_df.columns:
        succ_planned_cols = ['succ_task_planned_days']
    
    # Get all predecessor planned duration columns
    pred_planned_cols = [col for col in deps_df.columns if col.startswith('pred_task_planned_days_')]
    pred_planned_cols = sorted(pred_planned_cols, key=lambda x: int(x.split('_')[-1]))
    
    # Combine all feature columns in the same order as training
    feature_cols = pred_delay_cols + succ_planned_cols + pred_planned_cols
    
    print(f"Found {len(pred_delay_cols)} predecessor delay columns: {pred_delay_cols}")
    if succ_planned_cols:
        print(f"Found successor planned duration: {succ_planned_cols}")
    print(f"Found {len(pred_planned_cols)} predecessor planned duration columns: {pred_planned_cols}")
    print(f"Total features: {len(feature_cols)}")
    
    # Prepare feature matrix
    X_test = deps_df[feature_cols].copy()
    X_test = X_test.fillna(0)  # Fill NaN with 0 (tasks with fewer predecessors)
    
    print(f"\nFeature matrix shape: {X_test.shape}")
    print(f"Features statistics:")
    print(f"  Non-zero values per column (showing first 5):")
    for col in feature_cols[:5]:
        if col in X_test.columns:
            non_zero = (X_test[col] != 0).sum()
            print(f"    {col}: {non_zero:,} ({non_zero/len(X_test)*100:.1f}%)")
    
    # Make predictions
    print("\n" + "="*60)
    print("Generating Predictions")
    print("="*60)
    
    print(f"Predicting delays for {len(X_test):,} successor tasks...")
    predictions = predictor.predict(X_test)
    
    print(f"\nPrediction statistics:")
    print(f"  Mean: {predictions.mean():.3f} days")
    print(f"  Std: {predictions.std():.3f} days")
    print(f"  Min: {predictions.min():.3f} days")
    print(f"  Max: {predictions.max():.3f} days")
    
    # Add predictions to the dataframe
    # Find the position of succ_task_delay_days column
    cols = deps_df.columns.tolist()
    succ_delay_idx = cols.index('succ_task_delay_days')
    
    # Insert pred_succ_task_delay_days right after succ_task_delay_days
    deps_df.insert(succ_delay_idx + 1, 'pred_succ_task_delay_days', predictions)
    
    print(f"\nAdded predictions to dataframe.")
    print(f"New column order (showing first 10): {deps_df.columns.tolist()[:10]}")
    
    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    deps_df.to_parquet(output_file, index=False)
    
    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_file}")
    print(f"  Total rows: {len(deps_df):,}")
    print(f"  Total columns: {len(deps_df.columns)}")
    
    # Show comparison between actual and predicted delays
    if 'succ_task_delay_days' in deps_df.columns:
        actual = deps_df['succ_task_delay_days']
        predicted = deps_df['pred_succ_task_delay_days']
        
        # Calculate metrics for non-null actual values
        mask = actual.notna()
        if mask.sum() > 0:
            actual_valid = actual[mask]
            predicted_valid = predicted[mask]
            
            mae = np.abs(actual_valid - predicted_valid).mean()
            rmse = np.sqrt(((actual_valid - predicted_valid) ** 2).mean())
            
            print(f"\nModel Performance (on test data with actual delays):")
            print(f"  MAE: {mae:.3f} days")
            print(f"  RMSE: {rmse:.3f} days")
            print(f"  Samples with actual delays: {mask.sum():,} / {len(deps_df):,}")
    
    # Show sample
    print(f"\nSample predictions (first 5 rows, showing key columns):")
    sample_cols = ['project_id', 'succ_task_id', 'succ_task_delay_days', 'succ_task_planned_days',
                   'pred_succ_task_delay_days', 'num_predecessors']
    pred_cols_to_show = [col for col in deps_df.columns if 'pred_task_delay_days_' in col or 'pred_task_planned_days_' in col][:6]
    print(deps_df[sample_cols + pred_cols_to_show].head(5).to_string(index=False))


if __name__ == '__main__':
    main()
