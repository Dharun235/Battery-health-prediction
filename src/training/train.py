"""Model training loop utilities."""

import torch
import torch.nn as nn
from typing import Tuple
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train model for one epoch.
    
    Performs forward pass, computes loss, backward pass, and parameter update
    for all batches in the dataloader.
    
    Args:
        model: PyTorch model in training mode.
        dataloader: Training dataloader.
        optimizer: Optimizer for parameter updates.
        loss_fn: Loss function.
        device: Device (cpu/cuda).
    
    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in tqdm(dataloader, desc="Training", leave=False):
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float()
        
        optimizer.zero_grad()
        
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch.unsqueeze(1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Validate model on validation set.
    
    Evaluates model without gradient computation or parameter updates.
    
    Args:
        model: PyTorch model.
        dataloader: Validation dataloader.
        loss_fn: Loss function.
        device: Device (cpu/cuda).
    
    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Validating", leave=False):
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch.unsqueeze(1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0
