# **Machine Learning-based Prediction of Infarct Size in Patients with ST-segment Elevation Myocardial Infarction: A Multi-center Study**

<br>

## IS Prediction Pipeline

### Data Preprocessing
* The missing values (<1%) were imputed by the mice package in R. 
* Standardization of numerical variables were conducted. 

### Feature Selection
* Consider 3 combinations for all classifiers: 56 clinical features, 26 features selected by XGBoost (feature importance greater than average value), and the top10 important feautures by XGBoost. 
* Applied 5-fold cross validation. 

### Predictions
* We built a total of five ML models: random forest, light gradient boosting decision machine (LightGBM), deep forest, deep neural network, and stacking model. 
* Metrics used: MAE, $R^2$, $\epsilon$-Accuracy

## Files provided
* feature_selection.py: Functions for feature selection based on XGBoost F-score
* instantiate.py: A python script that define all the classifiers
* build_metric.py: Functions for choosing scoring and evaluating metrics
* stacking.py: Helper functions for building stacking ensemble model
* train_loop.py: A function to train specific type of regressor on different training feautures
* show_result.py: Helper functions to display evaluation metric or visualize the prediction result
* binary_case.py: Functions to calculate AUC and show result in binary classification case
* trained_models: Folder contains trained Random Forest regressors

















