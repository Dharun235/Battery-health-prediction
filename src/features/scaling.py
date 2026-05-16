"""Feature scaling utilities."""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def min_max_scale(
    data: Union[np.ndarray, pd.DataFrame],
    feature_range: Tuple[float, float] = (0, 1),
) -> Tuple[Union[np.ndarray, pd.DataFrame], dict]:
    """Scale features to a specific range using min-max normalization.
    
    Scales each feature independently to a specified range [min, max].
    Formula: (x - min(x)) / (max(x) - min(x)) * (max_range - min_range) + min_range
    
    Args:
        data: Input data (array or DataFrame).
        feature_range: Target range (min_value, max_value). Default (0, 1).
    
    Returns:
        Tuple of (scaled_data, scaling_params) for inverse transform.
    
    Raises:
        ValueError: If feature_range is invalid.
    """
    if feature_range[0] >= feature_range[1]:
        raise ValueError(f"Invalid feature_range: {feature_range}")
    
    is_dataframe = isinstance(data, pd.DataFrame)
    data_array = data.values if is_dataframe else data
    
    mins = data_array.min(axis=0)
    maxs = data_array.max(axis=0)
    
    min_range, max_range = feature_range
    scaled = (data_array - mins) / (maxs - mins + 1e-8) * (max_range - min_range) + min_range
    
    if is_dataframe:
        scaled = pd.DataFrame(scaled, columns=data.columns, index=data.index)
    
    params = {"mins": mins, "maxs": maxs, "range": feature_range}
    return scaled, params


def standard_scale(
    data: Union[np.ndarray, pd.DataFrame],
) -> Tuple[Union[np.ndarray, pd.DataFrame], dict]:
    """Standardize features to zero mean and unit variance (Z-score).
    
    Formula: (x - mean(x)) / std(x)
    
    Args:
        data: Input data (array or DataFrame).
    
    Returns:
        Tuple of (scaled_data, scaling_params) for inverse transform.
    """
    is_dataframe = isinstance(data, pd.DataFrame)
    data_array = data.values if is_dataframe else data
    
    means = data_array.mean(axis=0)
    stds = data_array.std(axis=0)
    
    scaled = (data_array - means) / (stds + 1e-8)
    
    if is_dataframe:
        scaled = pd.DataFrame(scaled, columns=data.columns, index=data.index)
    
    params = {"means": means, "stds": stds}
    return scaled, params
