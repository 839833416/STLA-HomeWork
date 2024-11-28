import numpy as np
import pandas as pd
import os

# 设置数据文件路径
data_dir = './SLTA_Projects/regression' # 替换为实际的文件路径
train_file = os.path.join(data_dir, 'regression_train.csv')
valid_file = os.path.join(data_dir, 'regression_val.csv')
test_file = os.path.join(data_dir, 'regression_test.csv')

# Load datasets
train_data = pd.read_csv(train_file)
valid_data = pd.read_csv(valid_file)
test_data = pd.read_csv(test_file)

# Separate features and labels
X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_valid, y_valid = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]

# Normalize features (optional, improves convergence for gradient descent)
X_train = (X_train - X_train.mean()) / X_train.std()
X_valid = (X_valid - X_train.mean()) / X_train.std()
X_test = (X_test - X_train.mean()) / X_train.std()

# Add bias term to the features
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_valid = np.c_[np.ones(X_valid.shape[0]), X_valid]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Train using Normal Equation
W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Validate the model
y_valid_pred = X_valid @ W
mse_valid = np.mean((y_valid - y_valid_pred) ** 2)
print(f"Validation MSE: {mse_valid}")

# Predict on the test set
y_test_pred = X_test @ W

# Save predictions
output = pd.DataFrame({'id': range(1, len(y_test_pred) + 1), 'y': y_test_pred})
output.to_csv('regression_test.csv', index=False)