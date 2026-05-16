"""Feature engineering - calendar and temporal features."""

import pandas as pd


def create_calendar_features(data: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """Extract calendar features from datetime column.
    
    Extracts temporal components like day, month, year, day of week, etc.
    Useful for capturing seasonal patterns and periodic behaviors.
    
    Args:
        data: Input DataFrame.
        datetime_col: Name of datetime column.
    
    Returns:
        DataFrame with calendar features added:
        - year, month, day, hour, minute
        - dayofweek (0=Monday, 6=Sunday)
        - quarter, dayofyear, weekofyear
    
    Raises:
        ValueError: If datetime column doesn't exist.
        TypeError: If column is not datetime.
    """
    if datetime_col not in data.columns:
        raise ValueError(f"Column '{datetime_col}' not found in DataFrame")
    
    result = data.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(result[datetime_col]):
        result[datetime_col] = pd.to_datetime(result[datetime_col])
    
    dt = result[datetime_col]
    
    result[f"{datetime_col}_year"] = dt.dt.year
    result[f"{datetime_col}_month"] = dt.dt.month
    result[f"{datetime_col}_day"] = dt.dt.day
    result[f"{datetime_col}_dayofweek"] = dt.dt.dayofweek
    result[f"{datetime_col}_quarter"] = dt.dt.quarter
    result[f"{datetime_col}_dayofyear"] = dt.dt.dayofyear
    result[f"{datetime_col}_weekofyear"] = dt.dt.isocalendar().week
    
    if dt.dt.hour is not None:
        result[f"{datetime_col}_hour"] = dt.dt.hour
        result[f"{datetime_col}_minute"] = dt.dt.minute
    
    return result
