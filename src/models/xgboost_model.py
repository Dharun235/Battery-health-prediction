"""XGBoost baseline model."""

import numpy as np
import xgboost as xgb
from typing import Tuple, Optional


class XGBoostBaseline:
    """XGBoost-based model for battery capacity prediction.
    
    Provides a gradient boosting baseline for comparison with deep learning models.
    Handles 1D targets by reshaping and flattening as needed.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ):
        """Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting stages. Default 100.
            max_depth: Maximum tree depth. Default 5.
            learning_rate: Learning rate (shrinkage). Default 0.1.
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=0,
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit XGBoost model.
        
        Args:
            X_train: Training features of shape (n_samples, n_features).
            y_train: Training targets of shape (n_samples,) or (n_samples, 1).
        """
        y_train_flat = y_train.ravel() if y_train.ndim > 1 else y_train
        self.model.fit(X_train, y_train_flat)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X_test: Test features of shape (n_samples, n_features).
        
        Returns:
            Predictions of shape (n_samples,).
        """
        return self.model.predict(X_test)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.
        
        Returns:
            Array of feature importance values.
        """
        return self.model.feature_importances_
