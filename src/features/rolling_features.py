"""Feature engineering - rolling window features."""

import pandas as pd
from typing import List


def create_rolling_features(
    data: pd.DataFrame,
    windows: List[int],
    target_column: str = None,
) -> pd.DataFrame:
    """Create rolling window features (moving average, std, etc).
    
    Computes moving statistics (mean, std, min, max) over specified windows.
    Captures short-term trends and volatility in time-series data.
    
    Args:
        data: Input DataFrame with time-series data.
        windows: List of window sizes (e.g., [7, 30, 90]).
        target_column: Specific column to compute rolling stats. If None, uses all numeric.
    
    Returns:
        DataFrame with original columns + rolling feature columns.
    
    Raises:
        ValueError: If windows contains non-positive integers.
    """
    if any(w <= 0 for w in windows):
        raise ValueError("All window sizes must be positive integers")
    
    result = data.copy()
    
    cols_to_process = [target_column] if target_column else data.select_dtypes(
        include=['number']
    ).columns.tolist()
    
    for col in cols_to_process:
        for window in windows:
            result[f"{col}_rolling_mean_{window}"] = data[col].rolling(window).mean()
            result[f"{col}_rolling_std_{window}"] = data[col].rolling(window).std()
            result[f"{col}_rolling_min_{window}"] = data[col].rolling(window).min()
            result[f"{col}_rolling_max_{window}"] = data[col].rolling(window).max()
    
    return result
