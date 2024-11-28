import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler  # 导入 StandardScaler

# 设置数据文件路径
data_dir = './SLTA_Projects/regression'  # 替换为实际的文件路径
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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Add bias term to the features
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_valid = np.c_[np.ones(X_valid.shape[0]), X_valid]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Define the linear regression model using gradient descent
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self
# 设置数据文件路径
data_dir = './SLTA_Projects/regression'  # 替换为实际的文件路径
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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Add bias term to the features
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_valid = np.c_[np.ones(X_valid.shape[0]), X_valid]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Define the linear regression model using gradient descent
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            self.weights -= self.learning_rate * dw

    def predict(self, X):
        return np.dot(X, self.weights)

# Initialize and train the model
lr_model = LinearRegression(learning_rate=0.01, n_iterations=1000)
lr_model.fit(X_train, y_train)

# Validate the model
y_valid_pred = lr_model.predict(X_valid)
mse_valid = np.mean((y_valid - y_valid_pred) ** 2)
print(f"Validation MSE: {mse_valid}")

# Predict on the test set
y_test_pred = lr_model.predict(X_test)

# Save predictions
output = pd.DataFrame({'id': range(1, len(y_test_pred) + 1), 'y': y_test_pred})
output.to_csv('regression_test.csv', index=False)