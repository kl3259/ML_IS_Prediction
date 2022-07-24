# **Machine Learning-based Prediction of Infarct Size in Patients with ST-segment Elevation Myocardial Infarction: A Multi-center Study**

<br>

## IS Prediction Pipeline

### Data Preprocessing
* Drop entries with missing rate >= 20%
* After dropping sparse entries, drop a predictor if its missing rate >= 50%
* Implement data imputation by MICE in R based on Random Forest

### Feature Selection
* Consider 3 combinations for all classifiers: all input features, 16 features selected by XGBoost, top10 important feautures by XGBoost
* Further examine top4 and top5 important predictors w/ best model found, exhaustive search all the combinations

### Predictions
* Regressor used: Random Forest, LightGBM, deep-forest, MLP, KNN, Stacking Ensemble
* Metrics used: MAE, $R^2$, $\epsilon$-Accuracy

## Files provided
* feature_selection.py: Functions for feature selection by XGBoost
* instantiate.py: A python script that define all the classifiers
* build_metric.py: Functions for choosing scoring and evaluating metrics
* stacking.py: Helper functions for building stacking ensemble model w/ some packages outside scikit-learn
* train_loop.py: A function to train specific type of regressor on different training feautures
* show_result.py: Helper functions to display evaluation metric or visualize the prediction result
* binary_case.py: Functions to calculate AUC and show result in binary classification case
* trained_models: Folder contains trained Random Forest regressors

















