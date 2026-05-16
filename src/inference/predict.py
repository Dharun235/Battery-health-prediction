"""Inference and prediction utilities."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Dict
from tqdm import tqdm


def predict_on_new_data(
    model: nn.Module,
    data: Union[np.ndarray, torch.Tensor],
    scaler: Optional[Dict] = None,
    device: torch.device = None,
) -> np.ndarray:
    """Make predictions on new data.
    
    Performs inference on new input data and optionally inverse transforms
    predictions using a provided scaler.
    
    Args:
        model: Trained PyTorch model.
        data: New input data of shape (n_samples, n_features).
        scaler: Scaler dictionary with 'mins' and 'maxs' for inverse transform.
        device: Device (cpu/cuda). Defaults to cuda if available.
    
    Returns:
        Predictions as numpy array.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    data = data.to(device)
    
    with torch.no_grad():
        predictions = model(data)
    
    predictions = predictions.cpu().numpy().flatten()
    
    # Inverse transform if scaler provided
    if scaler is not None:
        mins = scaler.get("mins", 0)
        maxs = scaler.get("maxs", 1)
        feature_range = scaler.get("range", (0, 1))
        min_range, max_range = feature_range
        
        predictions = (
            (predictions - min_range) / (max_range - min_range) * (maxs - mins) + mins
        )
    
    return predictions


def predict_batch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = None,
) -> np.ndarray:
    """Make batch predictions.
    
    Iterates through dataloader and makes predictions on all batches.
    
    Args:
        model: Trained PyTorch model.
        dataloader: Data loader with new samples.
        device: Device (cpu/cuda). Defaults to cuda if available.
    
    Returns:
        All predictions as numpy array.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    
    with torch.no_grad():
        for X_batch in tqdm(dataloader, desc="Predicting", leave=False):
            if isinstance(X_batch, (list, tuple)):
                X_batch = X_batch[0]
            
            X_batch = X_batch.to(device).float()
            predictions = model(X_batch)
            
            all_predictions.append(predictions.cpu().numpy().flatten())
    
    return np.concatenate(all_predictions) if all_predictions else np.array([])
