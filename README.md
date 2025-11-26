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

Authors
- Dharunkumar Senthilkumar
- Dhruvkumar Patel
