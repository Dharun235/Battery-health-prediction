# Battery Health Prediction: Time-Series Forecasting for Li-ion Battery Degradation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97-Model-blueviolet)](https://huggingface.co/Dharunkumar9/battery-capacity-predictor)

## Overview

This project implements state-of-the-art deep learning models for **time-series forecasting of Li-ion battery capacity degradation and remaining useful life (RUL) estimation**. Using voltage, current, and temperature measurements from charge-discharge cycles, our models predict battery health metrics critical for electric vehicle and energy storage applications.

**Key Achievement**: Transformer model achieves **0.0281 MAE** and **0.0186 MAPE** on battery capacity prediction tasks.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Dharun235/Battery-health-prediction.git
cd Battery-health-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Examples

```python
from src.data.load_data import load_nasa_dataset
from src.evaluation.metrics import mean_absolute_error

# Load dataset
data = load_nasa_dataset("dataset/NASA.npy")

# Use in your model
predictions = model(data)
mae = mean_absolute_error(data, predictions)
```

## Project Structure

```
.
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration & metadata
├── .gitignore                  # Git ignore rules
│
├── data/                       # Data directory
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Processed, cleaned datasets
│   └── NASA.npy                # Precomputed NASA dataset
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── models.ipynb            # LSTM, GRU, CNN-LSTM models
│   └── transformer.ipynb       # Transformer model
│
├── src/                        # Main source code
│   ├── data/                   # Data loading & preprocessing
│   │   ├── load_data.py        # Load .mat, .csv, .npy files
│   │   ├── clean.py            # Outlier removal, normalization
│   │   └── split.py            # Temporal train/test splitting
│   │
│   ├── features/               # Feature engineering
│   │   ├── lag_features.py     # Lag-based features
│   │   ├── rolling_features.py # Moving average features
│   │   ├── calendar_features.py# Time-based features
│   │   └── scaling.py          # Feature scaling utilities
│   │
│   ├── models/                 # Model architectures
│   │   ├── lstm.py             # LSTM implementation
│   │   ├── transformer.py      # Transformer implementation
│   │   ├── xgboost_model.py    # XGBoost baseline
│   │   └── baseline.py         # Statistical baselines
│   │
│   ├── training/               # Training utilities
│   │   ├── train.py            # Training loops
│   │   └── trainer.py          # High-level trainer class
│   │
│   ├── evaluation/             # Evaluation metrics & analysis
│   │   ├── metrics.py          # MAE, RMSE, MAPE, R²
│   │   ├── backtesting.py      # Backtesting framework
│   │   └── evaluate.py         # Evaluation pipeline
│   │
│   ├── inference/              # Inference & prediction
│   │   └── predict.py          # Prediction on new data
│   │
│   └── utils/                  # Utility functions
│       └── helpers.py          # Common utilities
│
├── configs/                    # Configuration files
│   ├── data.yaml               # Data configuration
│   ├── model.yaml              # Model hyperparameters
│   └── training.yaml           # Training configuration
│
├── outputs/                    # Generated outputs
│   └── plots/                  # Visualization plots
│
├── tests/                      # Unit tests
│   └── test_features.py        # Feature engineering tests
│
└── docs/                       # Documentation
    └── ARCHITECTURE.md         # Architecture & design decisions
```

## Models & Performance

Our research compared multiple architectures for battery RUL prediction:

| Model              | MAE        | RMSE       | MAPE       | Inference Speed |
| ------------------ | ---------- | ---------- | ---------- | --------------- |
| **Transformer** ⭐ | **0.0281** | **0.0349** | **0.0186** | ≈50ms           |
| GRU                | 0.0234     | 0.0430     | 0.0154     | ≈20ms           |
| CNN + LSTM         | 0.0313     | 0.0478     | 0.0206     | ≈35ms           |
| Basic LSTM         | 0.0442     | 0.0522     | 0.0299     | ≈30ms           |

**Transformer Model** (best overall accuracy):

- Available on [Hugging Face Model Hub](https://huggingface.co/Dharunkumar9/battery-capacity-predictor)
- 12 layers, 768 hidden dimensions, 12 attention heads
- Trained on 4 battery cells (B0005, B0006, B0007, B0018)

## Dataset

We use the **NASA Li-ion Battery Aging Dataset** featuring:

- **4 batteries**: B0005, B0006, B0007, B0018
- **Measurements**: Voltage, current, temperature per cycle
- **Labels**: Capacity degradation over time
- **Format**: MATLAB files (.mat) and CSV extracts

### Data Preparation

```python
from src.data.load_data import load_mat_file
from src.data.split import create_sliding_windows

# Load raw data
raw_data = load_mat_file("dataset/B0005.mat")

# Create sequences for model input
X, y = create_sliding_windows(data, sequence_length=100, forecast_horizon=10)
```

## Usage Examples

### Basic Workflow

```python
import torch
from src.data.load_data import load_csv_data
from src.data.clean import normalize_features
from src.data.split import temporal_train_test_split, create_sliding_windows
from src.evaluation.metrics import mean_absolute_error, root_mean_squared_error

# 1. Load data
data = load_csv_data("data/raw/B0005_charge.csv")

# 2. Clean and normalize
data_clean, params = normalize_features(data, method="minmax")

# 3. Split temporally
train_data, test_data = temporal_train_test_split(data_clean, train_size=0.8)

# 4. Create sliding windows
X_train, y_train = create_sliding_windows(train_data.values, sequence_length=100)
X_test, y_test = create_sliding_windows(test_data.values, sequence_length=100)

# 5. Train model
model = YourModel()
predictions = model(X_test)

# 6. Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
```

### Using Pre-trained Model

```python
from transformers import AutoModel
import torch

# Load pre-trained transformer from Hugging Face
model = AutoModel.from_pretrained("Dharunkumar9/battery-capacity-predictor")
model.eval()

# Make predictions
with torch.no_grad():
    output = model(input_tensor)
```

## Outputs & Results

The `outputs/` directory stores all generated artifacts (not versioned in git):

```
outputs/
├── checkpoints/        # Saved model weights (.pt files)
│   ├── best_model.pt              # Best model during training
│   ├── model_lstm_v1.pt           # LSTM variant checkpoint
│   └── model_transformer_v1.pt    # Transformer variant checkpoint
│
├── forecasts/          # Prediction results (.csv files)
│   ├── predictions_test.csv       # Test set predictions
│   ├── predictions_full.csv       # Full dataset predictions
│   └── residuals.csv              # Prediction errors
│
├── logs/               # Training logs and metrics
│   ├── training_log.txt           # Training metrics and loss history
│   └── tensorboard/               # TensorBoard event files
│
└── plots/              # Visualization outputs
    ├── dataset.png                # Dataset overview visualization
    ├── B0018_transformer.png      # Transformer model results
    └── B0018_other_models.png     # Other models comparison
```

Example usage:

```python
from src.training.trainer import Trainer

# Train and save checkpoint
trainer.fit(train_loader, val_loader)
trainer.save_checkpoint("outputs/checkpoints/model_v1.pt")

# Load for inference
trainer.load_checkpoint("outputs/checkpoints/model_v1.pt")
predictions = model(test_data)
```

## Notebooks

Interactive Jupyter notebooks for exploration and experimentation:

- **[models.ipynb](notebooks/models.ipynb)** - LSTM, GRU, CNN-LSTM models
  - Data loading and preprocessing
  - Model architectures and training
  - Evaluation and results visualization

- **[transformer.ipynb](notebooks/transformer.ipynb)** - Transformer model (best performer)
  - Transformer architecture implementation
  - Training procedures
  - Performance comparison with other models
  - Predictions vs actual plots

To run notebooks:

```bash
jupyter notebook notebooks/
# or
jupyter lab notebooks/
```

## Complete Training Example

Here's a complete pipeline using the modular src code:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_data import load_csv_data
from src.data.clean import normalize_features
from src.data.split import temporal_train_test_split, create_sliding_windows
from src.models.transformer import TransformerModel
from src.training.trainer import Trainer
from src.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from src.utils.helpers import set_random_seed, get_device

# Set seed for reproducibility
set_random_seed(42)
device = get_device()

# 1. Load and preprocess data
data = load_csv_data("data/raw/B0005_charge.csv")
data_norm, params = normalize_features(data, method="minmax")

# 2. Temporal split
train_data, test_data = temporal_train_test_split(data_norm, train_size=0.8)

# 3. Create sequences
X_train, y_train = create_sliding_windows(train_data.values, sequence_length=100)
X_test, y_test = create_sliding_windows(test_data.values, sequence_length=100)

# 4. Create dataloaders
train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).float()
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# 5. Initialize model, optimizer, loss
model = TransformerModel(
    input_dim=X_train.shape[2],
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    feedforward_dim=3072,
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 6. Train with Trainer class
config = {
    "num_epochs": 100,
    "early_stopping_patience": 20,
    "save_best_model": True,
    "best_model_path": "outputs/checkpoints/best_model.pt",
}

trainer = Trainer(model, optimizer, loss_fn, config, device)
history = trainer.fit(train_loader, train_loader)  # Use val_loader in practice

# 7. Evaluate
X_test_tensor = torch.from_numpy(X_test).float()
with torch.no_grad():
    predictions = model(X_test_tensor.to(device)).cpu().numpy()

mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# 8. Save predictions to outputs
import pandas as pd
results = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions.flatten(),
    "error": y_test - predictions.flatten(),
})
results.to_csv("outputs/forecasts/predictions_test.csv", index=False)
```

## Key Features

✅ **Clean, maintainable codebase** following SOLID principles  
✅ **Comprehensive data pipeline** (loading → cleaning → feature engineering)  
✅ **Multiple model architectures** (LSTM, GRU, Transformer, XGBoost)  
✅ **Robust evaluation framework** with standard metrics  
✅ **Jupyter notebooks** for exploration and experimentation  
✅ **Type hints** for code clarity and IDE support  
✅ **Well-documented functions** with clear docstrings  
✅ **Comprehensive test suite** (56+ test cases with pytest)

## Testing

Run the test suite to validate the implementation:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_features.py -v

# Run specific test class
pytest tests/test_features.py::TestLoadData -v
```

Test coverage includes:

- Data loading, cleaning, and preprocessing
- Feature engineering (lag, rolling, calendar, scaling)
- Train/test splitting with temporal integrity
- Evaluation metrics and backtesting
- Helper utilities

See [tests/README.md](tests/README.md) for detailed testing documentation.

## Development

### Code Quality Tools

```bash
# Format code with Black
black src/ tests/

# Check style with Flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# All-in-one check
black src/ tests/ && flake8 src/ tests/ && mypy src/
```

### Project Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or manually install dev tools
pip install black flake8 mypy pytest pytest-cov

# Create output directories
mkdir -p outputs/{checkpoints,forecasts,logs,plots}
```

## Future Work

- [ ] **Uncertainty Quantification**: Bayesian approaches for confidence intervals
- [ ] **Domain Adaptation**: Transfer learning across battery types
- [ ] **Real-time Monitoring**: Integration with IoT sensors
- [ ] **Explainability**: SHAP/LIME analysis for model interpretability
- [ ] **Federated Learning**: Privacy-preserving model training

## References

- **Dataset**: [Lithium-Ion Battery Aging Dataset](https://ieee-dataport.org/documents/lithium-ion-battery-data-set)
- **Transformer Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Course**: SSY340 - Deep Machine Learning, Chalmers University

## Citation

If you use this work, please cite:

```bibtex
@software{kumar2024battery,
  title={Battery Health Prediction: Time-Series Forecasting for Li-ion Battery Degradation},
  author={Kumar, Dharun and Patel, Dhruvkumar},
  year={2024},
  month={May},
  url={https://github.com/Dharun235/Battery-health-prediction},
  note={Course project: SSY340 - Deep Machine Learning, Chalmers University of Technology}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contact

- **Authors**: Dharun Kumar, Dhruvkumar Patel
- **GitHub**: [@Dharun235](https://github.com/Dharun235) | [@DhruvP301](https://github.com/DhruvP301)
- **Model Hub**: [Dharunkumar9 on Hugging Face](https://huggingface.co/Dharunkumar9)

---

**Made with ❤️ for better battery monitoring in the age of electric vehicles**
