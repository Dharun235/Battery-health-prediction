"""Data loading utilities for battery time-series datasets."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_mat_file(file_path: Path) -> Dict:
    """Load MATLAB .mat file containing battery cycle data.

    Args:
        file_path: Path to the .mat file.

    Returns:
        Dictionary containing the loaded MATLAB data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid .mat format.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        data = loadmat(file_path)
        return data
    except (OSError, ValueError) as error:
        raise ValueError(f"Failed to load .mat file {file_path}") from error


def load_nasa_dataset(file_path: Path) -> np.ndarray:
    """Load NASA battery dataset from .npy file.

    Args:
        file_path: Path to the .npy file.

    Returns:
        Numpy array containing the dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        data = np.load(file_path, allow_pickle=True)
        return data
    except (OSError, ValueError) as error:
        raise ValueError(f"Failed to load .npy file {file_path}") from error


def load_csv_data(file_path: Path) -> pd.DataFrame:
    """Load time-series data from CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If the CSV is malformed.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        data = pd.read_csv(file_path)
        return data
    except pd.errors.ParserError as error:
        raise ValueError(f"Failed to parse CSV file {file_path}") from error
