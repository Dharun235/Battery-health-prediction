"""Full evaluation pipeline."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Callable, Optional
from tqdm import tqdm


def evaluate_model(
    model: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    metrics: Optional[List[Callable]] = None,
    device: torch.device = None,
) -> Dict[str, float]:
    """Evaluate model on test set.
    
    Computes predictions on test set and evaluates using provided metrics.
    
    Args:
        model: Trained PyTorch model.
        test_dataloader: Test dataloader.
        metrics: List of metric functions. Each should accept (y_true, y_pred).
                If None, only computes MSE.
        device: Device (cpu/cuda). Defaults to cuda if available.
    
    Returns:
        Dictionary mapping metric names to computed values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_dataloader, desc="Evaluating", leave=False):
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            
            predictions = model(X_batch)
            
            all_predictions.append(predictions.cpu().numpy().flatten())
            all_actuals.append(y_batch.cpu().numpy().flatten())
    
    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_actuals)
    
    results = {}
    
    if metrics is None:
        # Default: compute MSE
        mse = np.mean((y_true - y_pred) ** 2)
        results["MSE"] = mse
    else:
        for metric in metrics:
            metric_name = getattr(metric, "__name__", "metric")
            results[metric_name] = metric(y_true, y_pred)
    
    return results
