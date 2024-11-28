import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 设置数据文件路径
data_dir = './SLTA_Projects/classification'  # 替换为实际的文件路径
train_file = os.path.join(data_dir, 'classification_train.csv')
valid_file = os.path.join(data_dir, 'classification_val.csv')
test_file = os.path.join(data_dir, 'classification_test.csv')

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

# Train using SVM
# Set hyper-parameters
C = 1.0  # Penalty factor
kernel = 'linear'  # Kernel function

svm_model = SVC(C=C, kernel=kernel)
svm_model.fit(X_train, y_train)

# Validate the model
y_valid_pred = svm_model.predict(X_valid)
accuracy_valid = accuracy_score(y_valid, y_valid_pred)
print(f"Validation Accuracy: {accuracy_valid}")

# Predict on the test set
y_test_pred = svm_model.predict(X_test)

# Save predictions
output = pd.DataFrame({'id': range(1, len(y_test_pred) + 1), 'y': y_test_pred})
output.to_csv('svm_classification_test.csv', index=False)