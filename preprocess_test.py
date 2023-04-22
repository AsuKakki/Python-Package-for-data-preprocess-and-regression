import pandas as pd
from preprocess import import_csv, handle_missing_values, visualize_outliers, normalize, discretize

file_path = "testfile.csv"
bins = 5

# Import the CSV file
df = import_csv(file_path)
print("Raw data:\n", df)

# Deal with missing values
df = handle_missing_values(df)
print("After processing the missing data:\n", df)

# Visualize anomalies
visualize_outliers(df)

# Normalize data
df_normalize = normalize(df)
print("After data normalize:\n", df)

# Discretize data
df_discretize = discretize(df, bins)
print("After data discretize:\n", df)

