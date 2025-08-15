# DRW Crypto Market Prediction – Kaggle 2025
This repository contains my work for the Kaggle DRW Crypto Market Prediction Competition (2025). The goal was to forecast short-term cryptocurrency market returns from noisy time-series data using anonymized features.

### Major Tools & Libraries
- Python, Pandas, NumPy
- LightGBM, XGBoost
- Optuna (hyperparameter tuning)
- Weights & Biases (experiment tracking)

### Approach
1. (01_setup_eda + 02_eda_2) I began with exploratory data analysis (and reducing the data to take up less space).
2. (03_lightgbm_baseline) I trained a baseline LightGBM model and compared the most important features to those from a Pearson correlation to the label. After that, I selected the top 400 features to use (reduced from 800+).
3. (04_lightgbm_timeseries) Then, I tweaked the LightGBM model a bit and ran it with time series splits (similar to K-fold analysis, but aware of the order of the data so there is no future data leakage when training).
4. (05_lightgbm_opt) Additionally, I used Optuna and wandb to track model performance with different hyperparameters to settle on the best hyperparameters to use and create predictions that way.
5. (06_xgboost_1) Finally, I trained an XGBoost model in a similar way.
6. For each run, I tracked rmse and Pearson performance, as the competition measured scores based on the Pearson correlation of submitted target predictions.
