"""Data cleaning and preprocessing utilities."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def remove_outliers(
    data: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Remove outliers from a specific column using IQR or Z-score method.

    Args:
        data: Input DataFrame.
        column: Column name to process.
        method: Outlier detection method ('iqr' or 'zscore'). Defaults to 'iqr'.
        threshold: IQR multiplier (default 1.5) or Z-score threshold (default 3).

    Returns:
        DataFrame with outliers removed.

    Raises:
        ValueError: If column does not exist or method is invalid.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if method == "iqr":
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    elif method == "zscore":
        from scipy import stats

        z_scores = np.abs(stats.zscore(data[column]))
        return data[z_scores < threshold]

    else:
        raise ValueError(f"Unknown method: {method}")


def fill_missing_values(
    data: pd.DataFrame,
    method: str = "forward",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fill missing values in DataFrame.

    Args:
        data: Input DataFrame with potential missing values.
        method: Filling method ('forward', 'backward', or 'interpolate'). Defaults to 'forward'.
        limit: Maximum number of consecutive NaN values to fill.

    Returns:
        DataFrame with missing values filled.

    Raises:
        ValueError: If method is invalid.
    """
    if method == "forward":
        return data.fillna(method="ffill", limit=limit)
    elif method == "backward":
        return data.fillna(method="bfill", limit=limit)
    elif method == "interpolate":
        return data.interpolate(limit=limit, method="linear")
    else:
        raise ValueError(f"Unknown fill method: {method}")


def normalize_features(
    data: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = "minmax",
) -> Tuple[pd.DataFrame, dict]:
    """Normalize specified columns to [0, 1] or zero-mean unit-variance.

    Args:
        data: Input DataFrame.
        columns: Columns to normalize. If None, normalizes all numeric columns.
        method: Normalization method ('minmax' or 'zscore'). Defaults to 'minmax'.

    Returns:
        Tuple of (normalized DataFrame, normalization parameters for inverse transform).

    Raises:
        ValueError: If method is invalid.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    normalized_data = data.copy()
    params = {}

    if method == "minmax":
        for col in columns:
            min_val = data[col].min()
            max_val = data[col].max()
            normalized_data[col] = (data[col] - min_val) / (max_val - min_val + 1e-8)
            params[col] = {"min": min_val, "max": max_val, "method": "minmax"}

    elif method == "zscore":
        for col in columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            normalized_data[col] = (data[col] - mean_val) / (std_val + 1e-8)
            params[col] = {"mean": mean_val, "std": std_val, "method": "zscore"}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_data, params
