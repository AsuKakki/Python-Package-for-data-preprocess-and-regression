# Python-Package-for-data-preprocess-and-regression

This is a Python package for basic data preprocessing and regression tasks. It provides easy-to-understand implementations of common data preprocessing tasks and linear and logistic regression algorithms.  

## Features of Modules  
### preprocess.py
Import CSV files：  
`
df = import_csv(file_path)
`  
Handle missing values:  
`
df = handle_missing_values(df)
`  
Anomaly detection through visualization:  
`
visualize_outliers(df)
`  
Data normalization:  
`
df_normalize = normalize(df)
`  
Data discretization:  
`
df_discretize = discretize(df, bins)
`  
### regression.py
Linear regression:  
`
coef_linear = regression.linear_regression(Feature matrix, Target vector)
`  
Logistic regression:  
`
coef_logistic = regression.logistic_regression(Feature matrix, Target vector, Learning rate, iterations)
`  
