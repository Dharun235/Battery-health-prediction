# Battery health prediction using machine learning models

This repository contains experiments and utilities for predicting battery capacity / remaining useful life (RUL) using deep learning models under the course SSY340 - Deep Machine Learning, Chalmers University.

## Project description
Developed deep learning models based on LSTM, GRU, CNN–LSTM, and Transformer architectures to predict Li-ion battery capacity degradation and estimate remaining useful life using time-series charge–discharge data. The codebase is implemented using python utilizing pytorch, leveraging voltage, current, and temperature trends with feature engineering and sliding time windows for temporal modeling. The Transformer achieved the best prediction accuracy, while GRU and CNN–LSTM provided efficient alternatives for smaller datasets. The work highlights effective data-driven approaches for battery health monitoring, critical for electric vehicles and energy storage systems. The plot of the capacity of all batteries used is shown below.
![Figure 1: Capacity degradation trends across cycles for all batteries.](https://github.com/Dharun235/Battery-health-prediction/blob/main/dataset.png)

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
![Figure 2: Predicted vs Actual capacity of B0018 for LSTM-CNN, GRU, LSTM based models](https://github.com/Dharun235/Battery-health-prediction/blob/main/B0018_other_models.png)

![Figure 3: Predicted vs Actual capacity of B0018 for transformer based based models](https://github.com/Dharun235/Battery-health-prediction/blob/main/B0018_transformer.png)

## Future work
- Integrate uncertainty quantification to improve capacity prediction reliability.
- Expand dataset diversity and explore domain adaptation to improve generalization.

## Models
Transformer model is saved here for future use - https://huggingface.co/Dharunkumar9/battery-capacity-predictor

## References
https://ieee-dataport.org/documents/lithium-ion-battery-data-set

### Authors
- Dharunkumar Senthilkumar
- Dhruvkumar Patel
