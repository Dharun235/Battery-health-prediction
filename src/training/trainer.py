"""High-level trainer class for model training."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
from .train import train_epoch, validate


class Trainer:
    """High-level trainer for battery capacity prediction models.
    
    Handles the entire training lifecycle including validation,
    checkpointing, and early stopping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: Dict,
        device: torch.device = None,
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            loss_fn: Loss function.
            config: Training configuration dictionary.
            device: Device (cpu/cuda). Defaults to cuda if available.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {"train_loss": [], "val_loss": []}
    
    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> Dict:
        """Train the model.
        
        Args:
            train_dataloader: Training dataloader.
            val_dataloader: Validation dataloader.
        
        Returns:
            Dictionary with training history.
        """
        num_epochs = self.config.get("num_epochs", 100)
        early_stopping_patience = self.config.get("early_stopping_patience", 20)
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = train_epoch(
                self.model,
                train_dataloader,
                self.optimizer,
                self.loss_fn,
                self.device,
            )
            
            # Validation phase
            val_loss = validate(
                self.model,
                val_dataloader,
                self.loss_fn,
                self.device,
            )
            
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if self.config.get("save_best_model", False):
                    self.save_checkpoint(
                        self.config.get("best_model_path", "best_model.pt")
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.training_history
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Checkpoint loaded from {path}")
