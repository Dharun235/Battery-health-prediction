# Battery health prediction using machine learning models

This repository contains experiments and utilities for predicting battery capacity / remaining useful life (RUL) using deep learning models under the course SSY340 - Deep Machine Learning, Chalmers University.

## Repository structure
- [models.ipynb](models.ipynb) — Contains code for data processing, model architecture, training and evaluation for LSTM, GRU and CNN-LSTM hybrid models.
- [transformer.ipynb](transformer.ipynb) — Contains code for data processing, model architecture, training and evaluation for transformer based model.
- [dataset/](dataset/) — original .mat files (B0005, B0006, B0007, B0018) and precomputed [NASA.npy](dataset/NASA.npy).
- [data/NASA/](data/NASA/) — raw CSVs extracted from [dataset/](dataset) used by the pipeline for LSTM, GRU and CNN-LSTM hybrid.

## Requirements
Install core Python packages used by the notebooks:
```sh
pip install numpy pandas scipy scikit-learn matplotlib torch transformers
```

## Typical workflow
- Load raw .mat from dataset/ for transformer model or CSV data from data/NASA/ for LSTM, GRU and CNN-LSTM hybrid models.
- Extract sequences with utility functions involving extracting capacity.
- Build sliding windows using build_instances.
- Train models in the notebooks; Models are saved in saved_models folder.

## Results
The table containing performance comparison across different deep learning models is shown below.
| Model        | MAE    | RMSE   | MAPE   |
|-------------|-------|-------|-------|
| Basic LSTM   | 0.0442 | 0.0522 | 0.0299 |
| Basic GRU    | 0.0234 | 0.0430 | 0.0154 |
| CNN + LSTM   | 0.0313 | 0.0478 | 0.0206 |
| Transformer  | 0.0281 | 0.0349 | 0.0186 |

and the plots of predicted vs actual capacity is shown below.
![Predicted vs Actual capacity of B0018 for LSTM-CNN, GRU, LSTM based models](https://github.com/Dharun235/Battery-health-prediction/blob/main/B0018_other_models.png)

![Predicted vs Actual capacity of B0018 for transformer based based models](https://github.com/Dharun235/Battery-health-prediction/blob/main/B0018_transformer.png)

## Future work
- Integrate uncertainty quantification to improve capacity prediction reliability.
- Expand dataset diversity and explore domain adaptation to improve generalization.

### Authors
- Dharunkumar Senthilkumar
- Dhruvkumar Patel
