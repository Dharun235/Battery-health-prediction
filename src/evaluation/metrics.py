"""Evaluation metrics for time-series forecasting."""

from typing import Union

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE score.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE score.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    Avoids division by zero by adding a small epsilon.

    Args:
        y_true: Ground truth values (should not be zero).
        y_pred: Predicted values.

    Returns:
        MAPE score (in percentage).

    Raises:
        ValueError: If all true values are zero.
    """
    if np.allclose(y_true, 0):
        raise ValueError("Cannot compute MAPE when all true values are zero")

    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))))


def r_squared_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        R² score (ranges from -∞ to 1, with 1 being perfect prediction).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / (ss_tot + 1e-8)))
