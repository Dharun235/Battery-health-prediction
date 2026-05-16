"""Data README - Dataset information and preparation guide."""

# Battery Health Dataset

## Files

- `NASA.npy`: Preprocessed NASA battery dataset (numpy format)
- Individual `.mat` files: Raw MATLAB data files for each battery
- `charge/` and `discharge/` subdirectories: CSV extracts by cycle type

## Data Preparation

1. **Raw Data** (NASA .mat files):
   - Load using `scipy.io.loadmat()`
   - Contains voltage, current, temperature, capacity measurements

2. **Processing Steps**:
   - Extract time-series features
   - Normalize to [0, 1] range
   - Create sliding windows for model input
   - Temporal train-test split (no shuffling)

3. **Output**:
   - `processed/` directory for cleaned datasets
   - Ready for model training

## Usage

```python
from src.data.load_data import load_nasa_dataset
data = load_nasa_dataset("dataset/NASA.npy")
```
