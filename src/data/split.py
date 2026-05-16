"""Data splitting utilities for train/validation/test sets."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def temporal_train_test_split(
    data: pd.DataFrame,
    train_size: float = 0.7,
    validation_size: Optional[float] = None,
    shuffle: bool = False,
) -> Tuple[pd.DataFrame, ...]:
    """Split time-series data into train/validation/test sets while preserving temporal order.

    For time-series data, we must not shuffle, as this violates the temporal dependency.

    Args:
        data: Input DataFrame or array.
        train_size: Proportion of data for training (0 to 1).
        validation_size: Proportion of data for validation. If None, no validation set.
        shuffle: Whether to shuffle data (should be False for time-series).

    Returns:
        Tuple of (train, validation, test) or (train, test) DataFrames.

    Raises:
        ValueError: If sizes don't sum to <= 1 or train_size is invalid.
    """
    if train_size <= 0 or train_size > 1:
        raise ValueError(f"train_size must be between 0 and 1, got {train_size}")

    total_size = train_size + (validation_size or 0)
    if total_size > 1:
        raise ValueError(
            f"train_size + validation_size cannot exceed 1, got {total_size}"
        )

    if shuffle:
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    n_samples = len(data)
    train_end = int(n_samples * train_size)

    train = data.iloc[:train_end]

    if validation_size is not None:
        validation_end = train_end + int(n_samples * validation_size)
        validation = data.iloc[train_end:validation_end]
        test = data.iloc[validation_end:]
        return train, validation, test
    else:
        test = data.iloc[train_end:]
        return train, test


def create_sliding_windows(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows for time-series prediction.

    Generates overlapping sequences of length `sequence_length` with corresponding
    target values `forecast_horizon` steps ahead.

    Args:
        data: Input time-series array of shape (n_timesteps, n_features).
        sequence_length: Number of timesteps in each window.
        forecast_horizon: Number of steps ahead to predict. Defaults to 1.
        stride: Step size between windows. Defaults to 1 (maximum overlap).

    Returns:
        Tuple of (X, y) where:
        - X: Array of shape (n_windows, sequence_length, n_features)
        - y: Array of shape (n_windows, forecast_horizon)

    Raises:
        ValueError: If sequence_length or forecast_horizon is invalid.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")

    min_length = sequence_length + forecast_horizon - 1
    if len(data) < min_length:
        raise ValueError(
            f"Data length ({len(data)}) is too short for sequence_length="
            f"{sequence_length} and forecast_horizon={forecast_horizon}"
        )

    windows_x = []
    windows_y = []

    for start_idx in range(0, len(data) - min_length + 1, stride):
        end_idx = start_idx + sequence_length
        target_idx = end_idx + forecast_horizon - 1

        window_x = data[start_idx:end_idx]
        window_y = data[target_idx]

        windows_x.append(window_x)
        windows_y.append(window_y)

    return np.array(windows_x), np.array(windows_y)
