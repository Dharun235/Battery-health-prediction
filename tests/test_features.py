"""Unit tests for data and feature engineering modules."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.load_data import load_csv_data
from src.data.clean import remove_outliers, fill_missing_values, normalize_features
from src.data.split import temporal_train_test_split, create_sliding_windows
from src.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r_squared_score,
)
from src.utils.helpers import set_random_seed, get_device, count_model_parameters


# Data Loading Tests

class TestLoadData:
    """Tests for data loading utilities."""

    def test_load_csv_data_valid_file(self):
        """Test loading a valid CSV file."""
        with TemporaryDirectory() as tmpdir:
            # Create a sample CSV
            csv_path = Path(tmpdir) / "test.csv"
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
            df.to_csv(csv_path, index=False)

            # Load it
            loaded_df = load_csv_data(csv_path)
            assert loaded_df.shape == (3, 2)
            assert list(loaded_df.columns) == ["col1", "col2"]

    def test_load_csv_data_file_not_found(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_csv_data(Path("nonexistent.csv"))

    def test_load_csv_data_empty_file(self):
        """Test loading an empty CSV file."""
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            df = pd.DataFrame()
            df.to_csv(csv_path, index=False)

            loaded_df = load_csv_data(csv_path)
            assert len(loaded_df) == 0


# Data Cleaning Tests

class TestRemoveOutliers:
    """Tests for outlier removal."""

    def test_remove_outliers_iqr_method(self):
        """Test IQR-based outlier removal."""
        # Create data with obvious outliers
        data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})
        cleaned = remove_outliers(data, "value", method="iqr", threshold=1.5)
        
        assert len(cleaned) < len(data)
        assert 100 not in cleaned["value"].values

    def test_remove_outliers_zscore_method(self):
        """Test Z-score based outlier removal."""
        data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})
        cleaned = remove_outliers(data, "value", method="zscore", threshold=3)
        
        assert len(cleaned) <= len(data)

    def test_remove_outliers_column_not_found(self):
        """Test error when column doesn't exist."""
        data = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            remove_outliers(data, "nonexistent", method="iqr")

    def test_remove_outliers_invalid_method(self):
        """Test error with invalid method."""
        data = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown method"):
            remove_outliers(data, "value", method="invalid")


class TestFillMissingValues:
    """Tests for missing value imputation."""

    def test_fill_missing_forward(self):
        """Test forward fill method."""
        data = pd.DataFrame({"value": [1.0, np.nan, 3.0, np.nan, 5.0]})
        filled = fill_missing_values(data, method="forward")
        
        assert not filled.isna().any().any()
        assert filled.iloc[1, 0] == 1.0
        assert filled.iloc[3, 0] == 3.0

    def test_fill_missing_backward(self):
        """Test backward fill method."""
        data = pd.DataFrame({"value": [1.0, np.nan, 3.0, np.nan, 5.0]})
        filled = fill_missing_values(data, method="backward")
        
        assert not filled.isna().any().any()
        assert filled.iloc[1, 0] == 3.0
        assert filled.iloc[3, 0] == 5.0

    def test_fill_missing_interpolate(self):
        """Test linear interpolation method."""
        data = pd.DataFrame({"value": [1.0, np.nan, 5.0]})
        filled = fill_missing_values(data, method="interpolate")
        
        assert not filled.isna().any().any()
        assert filled.iloc[1, 0] == 3.0  # Linear interpolation

    def test_fill_missing_invalid_method(self):
        """Test error with invalid method."""
        data = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
        with pytest.raises(ValueError, match="Unknown fill method"):
            fill_missing_values(data, method="invalid")


class TestNormalizeFeatures:
    """Tests for feature normalization."""

    def test_normalize_minmax(self):
        """Test min-max normalization."""
        data = pd.DataFrame({"col1": [0, 5, 10], "col2": [10, 20, 30]})
        normalized, params = normalize_features(data, method="minmax")
        
        assert normalized["col1"].min() >= 0
        assert normalized["col1"].max() <= 1
        assert "col1" in params
        assert params["col1"]["method"] == "minmax"

    def test_normalize_zscore(self):
        """Test Z-score normalization."""
        data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        normalized, params = normalize_features(data, method="zscore")
        
        assert abs(normalized["value"].mean()) < 1e-6  # Close to 0
        assert abs(normalized["value"].std() - 1.0) < 1e-6  # Close to 1

    def test_normalize_specific_columns(self):
        """Test normalizing only specific columns."""
        data = pd.DataFrame({"col1": [0, 10], "col2": [100, 200]})
        normalized, params = normalize_features(
            data, columns=["col1"], method="minmax"
        )
        
        assert "col1" in params
        assert "col2" not in params

    def test_normalize_invalid_method(self):
        """Test error with invalid method."""
        data = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_features(data, method="invalid")


# Data Splitting Tests

class TestTemporalTrainTestSplit:
    """Tests for temporal train-test splitting."""

    def test_temporal_split_basic(self):
        """Test basic temporal split."""
        data = pd.DataFrame({"value": range(100)})
        train, test = temporal_train_test_split(data, train_size=0.7)
        
        assert len(train) == 70
        assert len(test) == 30
        assert train.iloc[-1, 0] < test.iloc[0, 0]  # Temporal order

    def test_temporal_split_with_validation(self):
        """Test split with validation set."""
        data = pd.DataFrame({"value": range(100)})
        train, val, test = temporal_train_test_split(
            data, train_size=0.6, validation_size=0.2
        )
        
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_temporal_split_invalid_train_size(self):
        """Test error with invalid train size."""
        data = pd.DataFrame({"value": range(100)})
        with pytest.raises(ValueError):
            temporal_train_test_split(data, train_size=0)
        with pytest.raises(ValueError):
            temporal_train_test_split(data, train_size=1.5)

    def test_temporal_split_sizes_exceed_one(self):
        """Test error when sizes sum > 1."""
        data = pd.DataFrame({"value": range(100)})
        with pytest.raises(ValueError):
            temporal_train_test_split(
                data, train_size=0.7, validation_size=0.5
            )


class TestSlidingWindows:
    """Tests for sliding window creation."""

    def test_sliding_windows_basic(self):
        """Test basic sliding window creation."""
        data = np.random.randn(100, 5)
        X, y = create_sliding_windows(data, sequence_length=10, forecast_horizon=1)
        
        assert X.shape[0] == 90  # 100 - 10 = 90 windows
        assert X.shape[1] == 10  # sequence_length
        assert X.shape[2] == 5   # n_features
        assert y.shape == (90,)

    def test_sliding_windows_with_stride(self):
        """Test sliding windows with custom stride."""
        data = np.random.randn(100, 3)
        X, y = create_sliding_windows(
            data, sequence_length=10, forecast_horizon=1, stride=2
        )
        
        assert X.shape[0] == 45  # (100-10)/2 ≈ 45

    def test_sliding_windows_forecast_horizon(self):
        """Test sliding windows with forecast_horizon > 1."""
        data = np.arange(100).reshape(-1, 1)
        X, y = create_sliding_windows(
            data, sequence_length=10, forecast_horizon=5
        )
        
        assert X.shape[0] == 85  # 100 - 10 - 5 + 1 = 85
        assert y.shape == (85,)

    def test_sliding_windows_invalid_sequence_length(self):
        """Test error with invalid sequence_length."""
        data = np.random.randn(100, 5)
        with pytest.raises(ValueError):
            create_sliding_windows(data, sequence_length=0)

    def test_sliding_windows_data_too_short(self):
        """Test error when data is too short."""
        data = np.random.randn(5, 3)
        with pytest.raises(ValueError):
            create_sliding_windows(data, sequence_length=100, forecast_horizon=1)


# Evaluation Metrics Tests

class TestMetrics:
    """Tests for evaluation metrics."""

    def test_mae_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 0.0

    def test_mae_constant_error(self):
        """Test MAE with constant error."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 1.0

    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        rmse = root_mean_squared_error(y_true, y_pred)
        assert rmse == 0.0

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        
        rmse = root_mean_squared_error(y_true, y_pred)
        expected = np.sqrt(1/3)  # (0 + 0 + 1) / 3 = 1/3
        assert abs(rmse - expected) < 1e-6

    def test_mape_valid_calculation(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        mape = mean_absolute_percentage_error(y_true, y_pred)
        expected = 100 * np.mean([0.10, 0.05, 0.0333], axis=0)
        assert abs(mape - expected) < 0.1

    def test_mape_zero_true_values(self):
        """Test MAPE with zero true values raises error."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])
        
        with pytest.raises(ValueError):
            mean_absolute_percentage_error(y_true, y_pred)

    def test_r2_score_perfect_prediction(self):
        """Test R² score with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        r2 = r_squared_score(y_true, y_pred)
        assert r2 == 1.0

    def test_r2_score_constant_prediction(self):
        """Test R² score with mean prediction."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # Mean value
        
        r2 = r_squared_score(y_true, y_pred)
        assert abs(r2) < 1e-6  # Should be close to 0


# Utils Tests

class TestHelpers:
    """Tests for utility helper functions."""

    def test_set_random_seed(self):
        """Test random seed setting for reproducibility."""
        set_random_seed(42)
        rand1 = np.random.randn(5)
        
        set_random_seed(42)
        rand2 = np.random.randn(5)
        
        np.testing.assert_array_equal(rand1, rand2)

    def test_get_device(self):
        """Test device selection."""
        import torch
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]

    def test_get_device_type(self):
        """Test that device is either CPU or CUDA."""
        device = get_device()
        assert str(device) in ["cpu", "cuda"]


# Integration Tests

class TestIntegration:
    """Integration tests for full data pipeline."""

    def test_full_pipeline(self):
        """Test complete data processing pipeline."""
        with TemporaryDirectory() as tmpdir:
            # Create sample data
            csv_path = Path(tmpdir) / "data.csv"
            df = pd.DataFrame({
                "voltage": [3.0 + np.random.randn() * 0.1 for _ in range(100)],
                "current": [2.0 + np.random.randn() * 0.2 for _ in range(100)],
            })
            df.to_csv(csv_path, index=False)

            # Load data
            data = load_csv_data(csv_path)
            assert data.shape[0] == 100

            # Clean data
            data_clean = remove_outliers(data, "voltage", method="iqr")
            assert len(data_clean) > 0

            # Normalize data
            data_norm, params = normalize_features(data_clean, method="minmax")
            assert data_norm["voltage"].max() <= 1.0

            # Split data
            train, test = temporal_train_test_split(data_norm, train_size=0.7)
            assert len(train) + len(test) == len(data_norm)

            # Create windows
            X_train, y_train = create_sliding_windows(
                train.values, sequence_length=10
            )
            X_test, y_test = create_sliding_windows(
                test.values, sequence_length=10
            )
            
            assert X_train.shape[0] > 0
            assert X_test.shape[0] > 0
            assert X_train.shape[1] == 10  # sequence_length
            assert X_train.shape[2] == 2   # n_features
