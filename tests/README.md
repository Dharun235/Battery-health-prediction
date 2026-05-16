# Testing Guide

## Running Tests

### Install test dependencies

```bash
pip install pytest pytest-cov
```

### Run all tests

```bash
pytest tests/ -v
```

### Run with coverage report

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

### Run specific test file

```bash
pytest tests/test_features.py -v
```

### Run specific test class

```bash
pytest tests/test_features.py::TestMetrics -v
```

### Run specific test

```bash
pytest tests/test_features.py::TestMetrics::test_mae_perfect_prediction -v
```

## Test Structure

Tests are organized by module:

- **TestLoadData**: Data loading functionality
- **TestRemoveOutliers**: Outlier detection and removal
- **TestFillMissingValues**: Missing value imputation
- **TestNormalizeFeatures**: Feature normalization
- **TestTemporalTrainTestSplit**: Temporal data splitting
- **TestSlidingWindows**: Sliding window creation
- **TestMetrics**: Evaluation metrics (MAE, RMSE, MAPE, R²)
- **TestHelpers**: Utility functions
- **TestIntegration**: Full pipeline integration tests

## Coverage Goals

- **Target**: ≥ 80% code coverage
- **Critical paths**: Data pipeline, metrics, model utilities
- **Edge cases**: Invalid inputs, empty data, boundary conditions

## CI/CD Integration

Tests can be integrated into GitHub Actions:

```yaml
- name: Run tests
  run: pytest tests/ --cov=src --cov-report=xml
```

## Adding New Tests

When adding new features:

1. Write test cases before implementation (TDD)
2. Cover normal cases and edge cases
3. Use descriptive test names
4. Add docstrings explaining the test
5. Use fixtures for reusable data
6. Run tests locally before committing

Example:

```python
def test_new_feature_basic(self):
    """Test basic functionality of new feature."""
    # Setup
    data = create_test_data()

    # Execute
    result = new_feature(data)

    # Assert
    assert result.shape[0] > 0
    assert result is not None
```
