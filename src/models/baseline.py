"""Baseline and statistical models."""

import numpy as np
import pandas as pd
from typing import Union


def moving_average_baseline(
    data: Union[np.ndarray, pd.Series],
    window_size: int = 7,
) -> np.ndarray:
    """Simple moving average baseline for comparison.
    
    Predicts next value as the moving average of the window.
    Useful as a naive benchmark for time-series forecasting.
    
    Args:
        data: Input time-series (1D array or Series).
        window_size: Window size for moving average. Default 7.
    
    Returns:
        Predictions using moving average.
    
    Raises:
        ValueError: If window_size is invalid.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    
    if isinstance(data, pd.Series):
        data = data.values
    
    if len(data) < window_size:
        raise ValueError(
            f"Data length {len(data)} is smaller than window_size {window_size}"
        )
    
    predictions = []
    for i in range(window_size, len(data)):
        pred = np.mean(data[i - window_size : i])
        predictions.append(pred)
    
    return np.array(predictions)


def exponential_smoothing_baseline(
    data: Union[np.ndarray, pd.Series],
    alpha: float = 0.3,
) -> np.ndarray:
    """Exponential smoothing baseline for comparison.
    
    Simple exponential smoothing with smoothing parameter alpha.
    Formula: forecast_t = alpha * data_t + (1 - alpha) * forecast_{t-1}
    
    Args:
        data: Input time-series (1D array or Series).
        alpha: Smoothing coefficient (0 to 1). Default 0.3.
    
    Returns:
        Smoothed predictions.
    
    Raises:
        ValueError: If alpha is not in [0, 1].
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    if isinstance(data, pd.Series):
        data = data.values
    
    if len(data) < 2:
        raise ValueError("Data must have at least 2 points")
    
    smoothed = [data[0]]
    
    for i in range(1, len(data)):
        smooth_val = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(smooth_val)
    
    return np.array(smoothed)
