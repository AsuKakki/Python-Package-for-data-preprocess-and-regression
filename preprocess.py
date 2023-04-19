import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def import_csv(file_path):
    """
    Import CSV file

    param file_path:Path to the CSV file.
    returns:DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def handle_missing_values(df):
    """
    Deal with missing value. For numeric columns, it replaces
    missing values with the mean of the column. For categorical columns, it
    replaces missing values with the most frequent value in the column.

    param df:DataFrame with missing values.
    returns:DataFrame with missing values replaced.
    """
    for column in df.columns:
        # Check the type of columns
        if df[column].dtype == np.number:
            # Replace missing values with the mean of the column
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            # Replace missing values with the most frequent value in the column
            df[column].fillna(df[column].mode().iloc[0], inplace=True)
    return df

def visualize_outliers(df):
    """
    Creates boxplots for each numeric column in the DataFrame to visualize outliers

    param df:DataFrame containing the data
    return: None
    """
    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Check if the column is numeric
        if df[column].dtype == np.number:
            # Create a boxplot for the numeric column
            plt.figure()
            plt.boxplot(df[column])
            plt.title(f"Boxplot for {column}")
            plt.show()
    return

def normalize(df):
    """
    Normalizes numeric columns

    param df:DataFrame containing the data
    returns:DataFrame with numeric columns normalized.
    """
    for column in df.columns:
        # Check if the column is numeric
        if df[column].dtype == np.number:
            # Apply min-max scaling to normalize the numeric column
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

def discretize(df, bins):
    """
    Discretizes numeric columns

    param df: DataFrame containing the data.
    param bins : Number of equal-width bins to divide the value range into.
    returns: DataFrame with numeric columns discretized.
    """
    for column in df.columns:
        # Check if the column is numeric
        if df[column].dtype == np.number:
            # Discretize the numeric column by dividing its value range into equal-width bins
            df[column] = pd.cut(df[column], bins=bins, labels=False)
    return df
