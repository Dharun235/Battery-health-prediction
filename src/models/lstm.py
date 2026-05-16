"""LSTM model architecture."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM-based model for time-series forecasting.
    
    A simple LSTM architecture for predicting battery capacity degradation.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        """Initialize LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output
