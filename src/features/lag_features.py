"""Feature engineering - lag-based features for time-series."""

import pandas as pd
from typing import List, Union


def create_lag_features(
    data: pd.DataFrame,
    lag_values: List[int],
    target_column: str = None,
) -> pd.DataFrame:
    """Create lagged features for time-series prediction.
    
    Generates lagged versions of features at specified time lags.
    Useful for capturing temporal dependencies in time-series data.
    
    Args:
        data: Input DataFrame with time-series data.
        lag_values: List of lag periods (e.g., [1, 7, 30]).
        target_column: Specific column to lag. If None, lags all numeric columns.
    
    Returns:
        DataFrame with original columns + lag features appended.
    
    Raises:
        ValueError: If lag_values contains non-positive integers.
    """
    if any(lag <= 0 for lag in lag_values):
        raise ValueError("All lag values must be positive integers")
    
    result = data.copy()
    
    cols_to_lag = [target_column] if target_column else data.select_dtypes(
        include=['number']
    ).columns.tolist()
    
    for col in cols_to_lag:
        for lag in lag_values:
            result[f"{col}_lag_{lag}"] = data[col].shift(lag)
    
    return result
