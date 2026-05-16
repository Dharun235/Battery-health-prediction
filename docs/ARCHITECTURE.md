# Architecture & Design Decisions

## Overview

This document describes the architectural patterns and design decisions used in the Battery Health Prediction project.

## Design Principles

This project follows key software engineering principles:

### 1. **Clean Code** (Uncle Bob)

- Every function has a single responsibility
- Code reads like well-written prose
- Names are intention-revealing and unambiguous
- Functions are small (5-20 lines preferred)

### 2. **SOLID Principles**

- **Single Responsibility**: Each module handles one aspect (data loading, training, evaluation, etc.)
- **Open/Closed**: Models and trainers extend base classes without modification
- **Liskov Substitution**: Different model types can be swapped in training pipeline
- **Interface Segregation**: Modules only depend on the interfaces they use
- **Dependency Inversion**: Training pipeline depends on abstractions, not concrete models

### 3. **Temporal Integrity for Time-Series**

Since battery data is inherently temporal:

- **No shuffling** in train-test splits (see `src/data/split.py`)
- **Sliding windows** respect temporal ordering
- **Data leakage prevention**: Future data never influences past predictions

## Module Organization

```
src/
├── data/           # Data pipeline (load, clean, split)
├── features/       # Feature engineering (transformations)
├── models/         # Model architectures (LSTM, Transformer, etc.)
├── training/       # Training loops and utilities
├── evaluation/     # Metrics and evaluation
├── inference/      # Prediction on new data
└── utils/          # Shared utilities
```

### Separation of Concerns

**Data Module** (`src/data/`)

- `load_data.py`: File I/O (agnostic to content)
- `clean.py`: Data preprocessing
- `split.py`: Temporal train/test splitting

**Benefits**:

- Easy to swap file formats without changing preprocessing
- Preprocessing is independent of data source
- Train/test split logic is reusable

**Models Module** (`src/models/`)

- Each architecture in its own file
- Consistent interface for all models
- Inheritance from `torch.nn.Module` for compatibility

**Training Module** (`src/training/`)

- `train.py`: Low-level training loops (flexible)
- `trainer.py`: High-level trainer class (convenient)

## Key Design Patterns

### 1. **Pipeline Pattern** (Data Processing)

```python
# Data flows through stages: Load → Clean → Normalize → Split
data = load_csv_data(path)
data = remove_outliers(data)
data, params = normalize_features(data)
train, test = temporal_train_test_split(data)
```

Benefits:

- Easy to add new preprocessing steps
- Each step is testable independently
- Clear data transformations

### 2. **Strategy Pattern** (Models)

Different forecasting strategies (LSTM, Transformer, XGBoost) implement the same interface:

```python
class BaseModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

class LSTMModel(BaseModel):
    def forward(self, x: Tensor) -> Tensor:
        # LSTM-specific implementation

class TransformerModel(BaseModel):
    def forward(self, x: Tensor) -> Tensor:
        # Transformer-specific implementation
```

Benefits:

- Models are interchangeable in training loop
- New models can be added without changing trainer
- Easy to compare architectures

### 3. **Configuration Pattern** (YAML files)

```yaml
# configs/model.yaml
model:
  type: "transformer"
  params:
    hidden_dim: 768
    num_layers: 12
    num_heads: 12
```

Benefits:

- Hyperparameters decoupled from code
- Easy to run experiments with different configs
- Reproducibility without code changes

## Data Flow

```
Raw Data (MATLAB/CSV)
    ↓
    [load_data.load_mat_file / load_csv_data]
    ↓
Pandas DataFrame
    ↓
    [clean.remove_outliers, clean.fill_missing_values]
    ↓
Cleaned Data
    ↓
    [clean.normalize_features]
    ↓
Normalized Data
    ↓
    [split.temporal_train_test_split]
    ↓
[Train Data] [Test Data]
    ↓
    [split.create_sliding_windows]
    ↓
[Sequences] [Targets]
    ↓
    [PyTorch DataLoader]
    ↓
Model Training / Evaluation
```

## Error Handling Strategy

- **Fail Fast**: Validate inputs immediately
- **Clear Messages**: Exceptions explain what went wrong
- **Type Hints**: Reduce type-related errors

Example:

```python
def create_sliding_windows(data, sequence_length, ...):
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    # ...
```

## Testing Strategy

- **Unit Tests**: Test individual functions (`src/data/`, `src/features/`)
- **Integration Tests**: Test full pipelines
- **No fixtures needed**: Functions are pure (deterministic, no side effects)

Example test structure:

```python
# tests/test_features.py
def test_normalize_features_minmax():
    data = create_sample_data()
    normalized, params = normalize_features(data, method="minmax")
    assert normalized.min() >= 0
    assert normalized.max() <= 1

def test_create_sliding_windows_length():
    data = np.random.randn(1000, 5)
    X, y = create_sliding_windows(data, sequence_length=50)
    assert X.shape == (951, 50, 5)  # (n_windows, seq_len, features)
```

## Type Hints

All public functions use type hints:

```python
def normalize_features(
    data: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = "minmax",
) -> Tuple[pd.DataFrame, dict]:
    """..."""
```

Benefits:

- IDE auto-completion
- Static type checking with mypy
- Self-documenting code
- Catch type errors early

## Performance Considerations

1. **Vectorization**: Use NumPy/Pandas operations (not Python loops)
2. **Lazy Loading**: Load data only when needed
3. **GPU Support**: PyTorch models use CUDA when available
4. **Memory Efficient**: Generator-based DataLoaders for large datasets

Example:

```python
# Fast: vectorized
normalized = (data - data.min()) / (data.max() - data.min())

# Slow: element-wise
normalized = pd.DataFrame([(v - data.min()) / (data.max() - data.min())
                           for v in data])
```

## Future Enhancements

1. **Config Validation**: Use Pydantic for type-safe config
2. **Logging**: Structured logging for debugging
3. **Caching**: Cache preprocessing results
4. **Parallelization**: Parallelize feature engineering

---

**Last Updated**: 2024
**Author**: Dharun Kumar
