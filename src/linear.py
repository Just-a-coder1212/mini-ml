
import numpy as np
import pandas as pd
import time

# 1. Load the dataset (outside the training timer)
df = pd.read_csv("linear_regression_1m.csv")

data = df.to_numpy(dtype=np.float32)

X = data[:, :-1]
y = data[:, -1]

# 2. Train-test split without shuffling
train_size = int(round(len(X) * 0.80))

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

# 3. Initialize parameters
n_features = X.shape[1]

weights = np.zeros(n_features, dtype=np.float32)
bias = 0.0

lr = 0.001
epochs = 1000
m = len(X_train)

# 4. Train the model
start_time = time.perf_counter()

for epoch in range(epochs):
    # Predictions for all training samples
    y_pred = X_train @ weights + bias

    # Residuals
    errors = y_pred - y_train

    # Gradients of mean squared error
    dw = (X_train.T @ errors) / m
    db = np.mean(errors)

    # Update parameters
    weights -= lr * dw
    bias -= lr * db

end_time = time.perf_counter()

# 5. Evaluate on the test set
y_test_pred = X_test @ weights + bias

mse = np.mean((y_test - y_test_pred) ** 2)

print("\nTraining finished successfully!")
print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
print(f"Time taken: {end_time - start_time:.4f} seconds")
print(f"Test MSE: {mse:.6f}")
print(f"Intercept: {bias:.6f}")
print("Weights:", weights)
