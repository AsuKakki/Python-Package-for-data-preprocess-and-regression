import numpy as np
import regression

# Generate data for test
np.random.seed(50)
X_linear = np.random.randn(100, 2)
y_linear = np.random.randn(100)

np.random.seed(100)
X_logistic = np.random.randn(100, 2)
y_logistic = (np.random.randn(100) > 0).astype(int)

# Perform linear regression
coef_linear = regression.linear_regression(X_linear, y_linear)
print("Linear regression coefficients:", coef_linear)

# Perform logistic regression
coef_logistic = regression.logistic_regression(X_logistic, y_logistic, alpha=0.1, iterations=1000)
print("Logistic regression coefficients:", coef_logistic)