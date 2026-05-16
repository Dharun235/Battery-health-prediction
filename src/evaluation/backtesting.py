"""Backtesting framework for time-series models."""

import numpy as np
from typing import Dict, Tuple


def walk_forward_backtest(
    model,
    data: np.ndarray,
    train_size: int,
    test_size: int,
    step_size: int = 1,
) -> Dict:
    """Perform walk-forward backtesting on time-series data.
    
    Implements walk-forward validation: iteratively train on historical data
    and test on future data, moving the window forward each iteration.
    Respects temporal ordering and prevents data leakage.
    
    Args:
        model: Model with fit() and predict() methods.
        data: Full time-series dataset of shape (n_samples, n_features).
        train_size: Size of training window.
        test_size: Size of test window.
        step_size: Steps to move window forward. Default 1.
    
    Returns:
        Dictionary with keys:
        - 'predictions': All predictions
        - 'actuals': All actual values
        - 'indices': Indices of predictions in original data
    
    Raises:
        ValueError: If window sizes are invalid.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    
    if train_size + test_size > len(data):
        raise ValueError(
            f"train_size + test_size ({train_size + test_size}) "
            f"exceeds data length ({len(data)})"
        )
    
    all_predictions = []
    all_actuals = []
    all_indices = []
    
    # Walk forward through the data
    for i in range(0, len(data) - train_size - test_size, step_size):
        # Split data
        train_data = data[i : i + train_size]
        test_data = data[i + train_size : i + train_size + test_size]
        
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        
        # Train and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Store results
        all_predictions.extend(predictions.flatten() if predictions.ndim > 1 else predictions)
        all_actuals.extend(y_test.flatten())
        all_indices.extend(
            range(i + train_size, i + train_size + test_size)
        )
    
    return {
        "predictions": np.array(all_predictions),
        "actuals": np.array(all_actuals),
        "indices": np.array(all_indices),
    }
