import numpy as np
import pandas as pd

def linear_regression(X, y):
    """
    Performs linear regression

    param X : Feature matrix (m x n).
    param y : Target vector (m x 1).
    returns: Coefficients of the linear regression model.
    """
    # Add a column of ones to the feature matrix X
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Compute the coefficients using the normal equation
    coef = np.linalg.inv(X.T @ X) @ X.T @ y

    return coef

def sigmoid(z):
    """
    Computes the sigmoid function for the given input.

    param z :Input value(s) for the sigmoid function.
    returns: The sigmoid function value(s) for the input value(s).
    """
    # Compute the sigmoid function value(s) for the input value(s)
    sigmoid_value = 1 / (1 + np.exp(-z))
    return sigmoid_value

def logistic_regression(X, y, alpha=0.01, iterations=1000):
    """
    Performs logistic regression

    param X : Feature matrix (m x n).
    param y : Target vector (m x 1).
    alpha : Learning rate for gradient descent.
    iterations :Number of iterations for gradient descent.

    Returns: Coefficients of the logistic regression model (n+1 x 1).
    """
    # Get the number of samples (m) and features (n)
    m, n = X.shape

    # Add a column of ones to the feature matrix X for the intercept term
    X = np.hstack([np.ones((m, 1)), X])

    # Initialize the coefficients with random values
    coef = np.random.randn(n + 1)

    # Perform gradient descent for logistic regression
    for i in range(iterations):
        # Compute the predicted probabilities using the sigmoid function
        z = X @ coef
        y_pred = sigmoid(z)

        # Update the coefficients using the gradient and learning rate
        coef -= alpha * (X.T @ (y_pred - y)) / m

    return coef
