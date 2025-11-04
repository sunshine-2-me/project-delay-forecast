"""
Train a model to predict successor task delays from predecessor task delays.

Uses the dependencies_with_predecessor_counts.parquet file created by
dependency_analysis/prepare_train_dependencies.py.

Target: succ_task_delay_days
Features: 
  - pred_task_delay_days_1, pred_task_delay_days_2, pred_task_delay_days_3, ...
  - succ_task_planned_days
  - pred_task_planned_days_1, pred_task_planned_days_2, pred_task_planned_days_3, ...
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib


class DelayPredictor:
    """Predicts successor task delays from predecessor task delays."""
    
    def __init__(self, model_type='lightgbm', random_state=42):
        """
        Initialize the predictor.
        
        Args:
            model_type: 'lightgbm' or 'xgboost'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        
        if self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'lightgbm' or 'xgboost'")
    
    def prepare_features(self, df):
        """
        Prepare features from the dependencies DataFrame.
        
        Args:
            df: DataFrame with columns including pred_task_delay_days_1, _2, _3, etc.
                and planned duration columns (succ_task_planned_days, pred_task_planned_days_1, etc.)
        
        Returns:
            X: Feature matrix
            y: Target vector
        """
        # Get all predecessor delay columns
        pred_delay_cols = [col for col in df.columns if col.startswith('pred_task_delay_days_')]
        pred_delay_cols = sorted(pred_delay_cols, key=lambda x: int(x.split('_')[-1]))
        
        # Get successor planned duration
        succ_planned_cols = []
        if 'succ_task_planned_days' in df.columns:
            succ_planned_cols = ['succ_task_planned_days']
        
        # Get all predecessor planned duration columns
        pred_planned_cols = [col for col in df.columns if col.startswith('pred_task_planned_days_')]
        pred_planned_cols = sorted(pred_planned_cols, key=lambda x: int(x.split('_')[-1]))
        
        # Combine all feature columns: delays first, then planned durations
        feature_cols = pred_delay_cols + succ_planned_cols + pred_planned_cols
        
        # Use these as features
        X = df[feature_cols].copy()
        y = df['succ_task_delay_days'].copy()
        
        # Fill NaN values with 0 (tasks with fewer predecessors)
        X = X.fillna(0)
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        return X, y
    
    def train(self, X, y, X_val=None, y_val=None, validation_split=0.2):
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            validation_split: Fraction to use for validation if X_val/y_val not provided
        """
        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"  Training samples: {len(X):,}")
        print(f"  Features: {len(X.columns)} ({', '.join(X.columns)})")
        
        # Split validation set if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
        
        print(f"  Training set: {len(X_train):,} samples")
        print(f"  Validation set: {len(X_val):,} samples")
        
        # Train model
        if self.model_type == 'lightgbm':
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:  # xgboost
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                early_stopping_rounds=50,
                verbose=False
            )
        
        # Evaluate
        print(f"\nModel Evaluation:")
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"  Training - MAE: {train_mae:.3f}, RMSE: {train_rmse:.3f}, R²: {train_r2:.3f}")
        print(f"  Validation - MAE: {val_mae:.3f}, RMSE: {val_rmse:.3f}, R²: {val_r2:.3f}")
        
        return {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Ensure same columns as training
        X_processed = X.copy()
        if self.feature_columns:
            # Fill missing columns with 0
            for col in self.feature_columns:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            # Reorder columns
            X_processed = X_processed[self.feature_columns]
        
        # Fill NaN values
        X_processed = X_processed.fillna(0)
        
        return self.model.predict(X_processed)
    
    def save(self, path):
        """Save the model and metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'random_state': self.random_state
        }, path)
        print(f"\nModel saved to {path}")
    
    def load(self, path):
        """Load the model and metadata."""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_columns = data['feature_columns']
        self.random_state = data.get('random_state', 42)
        print(f"Model loaded from {path}")


def main():
    """Main training function."""
    print("="*60)
    print("Training Delay Prediction Model")
    print("="*60)
    
    # Configuration
    data_file = 'out/train_dependencies.parquet'
    model_type = 'lightgbm'  # Change to 'xgboost' if preferred
    output_dir = Path('out/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {data_file}...")
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df):,} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing target values
    missing_target = df['succ_task_delay_days'].isna().sum()
    if missing_target > 0:
        print(f"\nWarning: {missing_target} samples have missing target values. Removing them...")
        df = df.dropna(subset=['succ_task_delay_days'])
        print(f"Remaining samples: {len(df):,}")
    
    # Prepare features and target
    predictor = DelayPredictor(model_type=model_type)
    X, y = predictor.prepare_features(df)
    
    print(f"\nTarget statistics:")
    print(f"  Mean: {y.mean():.3f}")
    print(f"  Std: {y.std():.3f}")
    print(f"  Min: {y.min():.3f}")
    print(f"  Max: {y.max():.3f}")
    
    # Train model
    metrics = predictor.train(X, y, validation_split=0.2)
    
    # Save model
    model_path = output_dir / f'delay_predictor_{model_type}.joblib'
    predictor.save(model_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"\nFinal Validation Metrics:")
    print(f"  MAE: {metrics['val_mae']:.3f} days")
    print(f"  RMSE: {metrics['val_rmse']:.3f} days")
    print(f"  R²: {metrics['val_r2']:.3f}")


if __name__ == '__main__':
    main()
