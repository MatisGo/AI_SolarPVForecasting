import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from itertools import product
from dnn_app_utils_v3 import *

# ========================================================================
# HYPERPARAMETERS - EASY TO CHANGE
# ========================================================================
LEARNING_RATE = 0.9                                    # Learning rate for gradient descent
ITERATION_TESTS = [500, 1000, 2000, 5000, 10000,      # Iteration numbers to test
                   25000, 50000, 100000, 250000, 500000]
LAYERS_DIMS = [7, 5, 3, 3, 5, 1]                      # Network architecture [input, hidden1, hidden2, ..., output]
PRINT_COST = False                                     # Print cost during training (False for cleaner output)
TEST_SIZE = 0.2                                        # Train/test split ratio
RANDOM_STATE = 42                                      # Random seed for reproducibility
# ========================================================================

# 1 - LOAD AND PREPROCESS DATA
print("="*70)
print("LOADING AND PREPROCESSING DATA")
print("="*70)

data = pd.read_excel('Data/RawData.xlsx')
data = data.dropna()  # Remove rows with missing values

# Extract hour from the first column (date), handling invalid dates
data['datetime'] = pd.to_datetime(data.iloc[:, 0], errors='coerce')
data = data.dropna(subset=['datetime'])  # Remove rows with invalid dates
data['hour'] = data['datetime'].dt.hour
data = data.drop(columns=['datetime'])  # Remove the datetime column

# Reorder columns to make hour the second column
first_col = data.columns[0]  # First feature column
hour_col = 'hour'
remaining_cols = [col for col in data.columns if col != first_col and col != hour_col and col != data.columns[-1]]

# Reorganize: first feature, hour, remaining features, target
data = data[[first_col, hour_col] + remaining_cols]

print(data.head())

# Separate features and target variable
feature_cols = list(data.columns[1:-2])  # All columns except the last (target)
X = data[feature_cols].values.astype(float)
y = data.iloc[:, -1].values.astype(float)   # Last column is target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Normalize features and target
scaler_X = MinMaxScaler(feature_range=(0,1))
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(0,1))
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Reshape for neural network (features, samples)
y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)
X_train = X_train.T
X_test = X_test.T

# Print dataset info
m_train = X_train.shape[1]
m_test = X_test.shape[1]
n_x = X_train.shape[0]
n_y = y_train.shape[0]

print("\n" + "="*70)
print("DATASET INFORMATION")
print("="*70)
print(f"Number of training examples: m_train = {m_train}")
print(f"Number of testing examples: m_test = {m_test}")
print(f"Number of features: n_x = {n_x}")
print(f"Number of output units: n_y = {n_y}")
print(f"train_set_x shape: {X_train.shape}")
print(f"train_set_y shape: {y_train.shape}")
print(f"test_set_x shape: {X_test.shape}")
print(f"test_set_y shape: {y_test.shape}")

# 2 - DEFINE L-LAYER MODEL FUNCTION
def L_layer_model(X, Y, layers_dims, learning_rate=0.85, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs -- list of costs recorded during training
    """

    np.random.seed(1)
    costs = []                         # keep track of cost

    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {np.squeeze(cost)}")
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

# 3 - RUN TESTS FOR DIFFERENT ITERATION COUNTS
print("\n" + "="*70)
print("RUNNING TESTS - VARIATION 1")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Architecture: {LAYERS_DIMS}")
print("="*70)

# Store results
results = []

for num_iter in ITERATION_TESTS:
    print(f"\n{'='*70}")
    print(f"Testing with {num_iter} iterations...")
    print(f"{'='*70}")

    # Start timing
    start_time = time.time()

    # Train the model
    parameters, costs = L_layer_model(X_train, y_train, LAYERS_DIMS,
                                      learning_rate=LEARNING_RATE,
                                      num_iterations=num_iter,
                                      print_cost=PRINT_COST)

    # End timing
    training_time = time.time() - start_time

    # Make predictions
    pred_train = predict(X_train, y_train, parameters)
    pred_test = predict(X_test, y_test, parameters)

    # Inverse transform to original scale
    y_pred_train_inv = scaler_y.inverse_transform(pred_train.flatten().reshape(-1, 1)).flatten()
    y_actual_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_pred_test_inv = scaler_y.inverse_transform(pred_test.flatten().reshape(-1, 1)).flatten()
    y_actual_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate metrics for TRAIN set
    mae_train = mean_absolute_error(y_actual_train_inv, y_pred_train_inv)
    mse_train = mean_squared_error(y_actual_train_inv, y_pred_train_inv)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_actual_train_inv, y_pred_train_inv)

    # Calculate metrics for TEST set
    mae_test = mean_absolute_error(y_actual_test_inv, y_pred_test_inv)
    mse_test = mean_squared_error(y_actual_test_inv, y_pred_test_inv)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_actual_test_inv, y_pred_test_inv)

    # Store results
    results.append({
        'Iterations': num_iter,
        'Test_R2': r2_test,
        'Test_MAE': mae_test,
        'Test_MSE': mse_test,
        'Test_RMSE': rmse_test,
        'Comput_Time_sec': training_time,
        'Train_R2': r2_train,
        'Train_MSE': mse_train,
        'Train_MAE': mae_train
    })

    print(f"Completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Test R²: {r2_test:.6f} | Train R²: {r2_train:.6f}")

# 4 - CREATE RESULTS TABLE
print("\n" + "="*70)
print("RESULTS TABLE - VARIATION 1")
print(f"Learning Rate: {LEARNING_RATE}")
print("="*70)

results_df = pd.DataFrame(results)

# Display the results table
print("\n" + results_df.to_string(index=False))

# 5 - SAVE RESULTS TO EXCEL
output_filename = f'DeepNN_Results_LR{LEARNING_RATE}.xlsx'
results_df.to_excel(output_filename, index=False)
print(f"\nResults saved to: {output_filename}")

# 6 - PLOT RESULTS
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: R² vs Iterations
axes[0, 0].plot(results_df['Iterations'], results_df['Test_R2'], 'o-', label='Test R²', color='blue')
axes[0, 0].plot(results_df['Iterations'], results_df['Train_R2'], 's-', label='Train R²', color='orange')
axes[0, 0].set_xlabel('Number of Iterations')
axes[0, 0].set_ylabel('R²')
axes[0, 0].set_title('R² Score vs Iterations')
axes[0, 0].legend()
axes[0, 0].grid(True)
axes[0, 0].set_xscale('log')

# Plot 2: MAE vs Iterations
axes[0, 1].plot(results_df['Iterations'], results_df['Test_MAE'], 'o-', label='Test MAE', color='blue')
axes[0, 1].plot(results_df['Iterations'], results_df['Train_MAE'], 's-', label='Train MAE', color='orange')
axes[0, 1].set_xlabel('Number of Iterations')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_title('MAE vs Iterations')
axes[0, 1].legend()
axes[0, 1].grid(True)
axes[0, 1].set_xscale('log')

# Plot 3: MSE vs Iterations
axes[1, 0].plot(results_df['Iterations'], results_df['Test_MSE'], 'o-', label='Test MSE', color='blue')
axes[1, 0].plot(results_df['Iterations'], results_df['Train_MSE'], 's-', label='Train MSE', color='orange')
axes[1, 0].set_xlabel('Number of Iterations')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('MSE vs Iterations')
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].set_xscale('log')

# Plot 4: Computation Time vs Iterations
axes[1, 1].plot(results_df['Iterations'], results_df['Comput_Time_sec'], 'o-', color='green')
axes[1, 1].set_xlabel('Number of Iterations')
axes[1, 1].set_ylabel('Computation Time (seconds)')
axes[1, 1].set_title('Computation Time vs Iterations')
axes[1, 1].grid(True)
axes[1, 1].set_xscale('log')

plt.tight_layout()
plt.savefig(f'DeepNN_Results_LR{LEARNING_RATE}.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("ALL TESTS COMPLETE!")
print("="*70)
print(f"\nBest Test R²: {results_df['Test_R2'].max():.6f} at {results_df.loc[results_df['Test_R2'].idxmax(), 'Iterations']} iterations")
print(f"Best Train R²: {results_df['Train_R2'].max():.6f} at {results_df.loc[results_df['Train_R2'].idxmax(), 'Iterations']} iterations")
print(f"Total computation time: {results_df['Comput_Time_sec'].sum():.2f} seconds ({results_df['Comput_Time_sec'].sum()/60:.2f} minutes)")
