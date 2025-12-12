# Rice Yield Estimation Using UAV and Machine Learning

This repository contains Python code for rice yield prediction using UAV-derived vegetation indices and machine learning models, which are Random Forest, XGBoost, Neural Networks, and Bayesian Model Averaging (BMA)

## Models Implemented are
- Random Forest
- XGBoost
- Neural Network
- Linear Regression
- BMA Full (all models)
- BMA Reduced (nonlinear ML models only)

## Contents
- `RP_ML_V21.py` – Main modeling and evaluation pipeline

## Methodology Overview
- UAV multispectral data preprocessing
- Vegetation index extraction across multiple growth stages
- Model training (300 iterations)
- Independent test set evaluation
- Ensemble modeling using BMA
- Performance assessment using benefit indices and error metrics

## Data Availability
Raw UAV imagery and field data are not included due to size and privacy constraints. The repository focuses on modeling workflows and evaluation methods.

## Status
This repository accompanies an unpublished manuscript currently under preparation. The code will be updated upon manuscript acceptance.

## Example Results figures contains
### Measured vs Predicted Yield scatterplot
### Model Performance Comparison barplots
### RMSE Distribution Across Models boxplots

## Author
Ritik Pokharel  
LSU |MS Student | Precision Agriculture 
