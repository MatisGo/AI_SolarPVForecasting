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
LEARNING_RATE = 0.9                                   # Learning rate for gradient descent (OPTIMAL for this problem)
ITERATION_TESTS = [ 100000, 200000, 500000, 1000000 ]  # Different iteration counts to test
LAYERS_DIMS = [7,40,40,40,40, 1]                      # Network architecture - SIMPLER & WIDER for better gradient flow
PRINT_COST = False                                     # Print cost during training (TRUE to see if it's decreasing!)
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
print("RUNNING MULTIPLE TESTS")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Architecture: {LAYERS_DIMS}")
print(f"Number of Tests: {len(ITERATION_TESTS)}")
print("="*70)

# Store results for all tests
all_results = []
total_start_time = time.time()

for test_num, num_iter in enumerate(ITERATION_TESTS, 1):
    print(f"\n{'#'*70}")
    print(f"### TEST {test_num}/{len(ITERATION_TESTS)}: {num_iter} ITERATIONS")
    print(f"{'#'*70}")

    # Start timing for this test
    start_time = time.time()

    # Train the model
    print(f"Training model with {num_iter} iterations...")
    parameters, costs = L_layer_model(X_train, y_train, LAYERS_DIMS,
                                      learning_rate=LEARNING_RATE,
                                      num_iterations=num_iter,
                                      print_cost=PRINT_COST)

    # End timing
    training_time = time.time() - start_time

    print(f"✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")

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
    all_results.append({
        'Iterations': num_iter,
        'Test_R2': r2_test,
        'Test_MAE': mae_test,
        'Test_MSE': mse_test,
        'Test_RMSE': rmse_test,
        'Train_R2': r2_train,
        'Train_MAE': mae_train,
        'Train_MSE': mse_train,
        'Train_RMSE': rmse_train,
        'Comput_Time_sec': training_time
    })

    # PRINT DETAILED RESUME FOR THIS TEST
    print("\n" + "="*70)
    print(f"RESUME - TEST {test_num}: {num_iter} ITERATIONS")
    print("="*70)
    print(f"{'Metric':<12} {'Train':>14} {'Test':>14} {'Diff':>14}")
    print("-" * 70)
    print(f"{'R²':<12} {r2_train:>14.6f} {r2_test:>14.6f} {r2_test - r2_train:>14.6f}")
    print(f"{'MAE':<12} {mae_train:>14.4f} {mae_test:>14.4f} {mae_test - mae_train:>14.4f}")
    print(f"{'MSE':<12} {mse_train:>14.4f} {mse_test:>14.4f} {mse_test - mse_train:>14.4f}")
    print(f"{'RMSE':<12} {rmse_train:>14.4f} {rmse_test:>14.4f} {rmse_test - rmse_train:>14.4f}")
    print("-" * 70)

    # Diagnostic
    if r2_train > 0.9 and r2_test < 0.7:
        diagnostic = "⚠️  OVERFITTING: High Train R², Low Test R²"
    elif r2_train < 0.5 and r2_test < 0.5:
        diagnostic = "⚠️  UNDERFITTING: Poor performance on Train and Test"
    elif abs(r2_train - r2_test) < 0.1:
        diagnostic = "✅ GOOD FIT: Similar performance on Train/Test"
    else:
        diagnostic = "⚠️  POSSIBLE OVERFITTING: Notable gap between Train and Test"

    print(f"Diagnostic: {diagnostic}")
    print(f"Computation Time: {training_time:.2f}s ({training_time/60:.2f} min)")
    print("="*70)

# Calculate total time
total_time = time.time() - total_start_time

# 4 - CREATE FINAL SUMMARY TABLE AND SAVE TO EXCEL
print("\n" + "#"*70)
print("### FINAL SUMMARY - ALL TESTS")
print("#"*70)

results_df = pd.DataFrame(all_results)

# Reorder columns for better readability
results_df = results_df[['Iterations',
                         'Test_R2', 'Test_MAE', 'Test_MSE', 'Test_RMSE',
                         'Train_R2', 'Train_MAE', 'Train_MSE', 'Train_RMSE',
                         'Comput_Time_sec']]

# Display the complete results table in console
print("\n" + results_df.to_string(index=False))

# Save to Excel with formatting
output_filename = f'DeepNN_Results_LR{LEARNING_RATE}_Arch{"_".join(map(str, LAYERS_DIMS))}.xlsx'

# Create Excel writer with xlsxwriter engine for formatting
with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
    # Write main results
    results_df.to_excel(writer, sheet_name='Results', index=False)

    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Results']

    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BD',
        'border': 1
    })

    number_format = workbook.add_format({'num_format': '#,##0.0000'})
    integer_format = workbook.add_format({'num_format': '#,##0'})
    time_format = workbook.add_format({'num_format': '#,##0.00'})

    # Set column widths and formats
    worksheet.set_column('A:A', 12, integer_format)  # Iterations
    worksheet.set_column('B:B', 12, number_format)   # Test_R2
    worksheet.set_column('C:C', 12, number_format)   # Test_MAE
    worksheet.set_column('D:D', 12, number_format)   # Test_MSE
    worksheet.set_column('E:E', 12, number_format)   # Test_RMSE
    worksheet.set_column('F:F', 12, number_format)   # Train_R2
    worksheet.set_column('G:G', 12, number_format)   # Train_MAE
    worksheet.set_column('H:H', 12, number_format)   # Train_MSE
    worksheet.set_column('I:I', 12, number_format)   # Train_RMSE
    worksheet.set_column('J:J', 16, time_format)     # Comput_Time_sec

    # Write headers with formatting
    for col_num, value in enumerate(results_df.columns.values):
        worksheet.write(0, col_num, value, header_format)

    # Add a summary sheet
    summary_df = pd.DataFrame({
        'Parameter': ['Learning Rate', 'Architecture', 'Test Size', 'Random State',
                      'Number of Tests', 'Total Computation Time (sec)', 'Total Computation Time (min)',
                      'Best Test R²', 'Best Test R² at Iterations',
                      'Best Train R²', 'Best Train R² at Iterations'],
        'Value': [LEARNING_RATE, str(LAYERS_DIMS), TEST_SIZE, RANDOM_STATE,
                  len(ITERATION_TESTS), f'{total_time:.2f}', f'{total_time/60:.2f}',
                  f'{results_df["Test_R2"].max():.6f}', results_df.loc[results_df["Test_R2"].idxmax(), "Iterations"],
                  f'{results_df["Train_R2"].max():.6f}', results_df.loc[results_df["Train_R2"].idxmax(), "Iterations"]]
    })

    summary_df.to_excel(writer, sheet_name='Summary', index=False)

    # Format summary sheet
    summary_sheet = writer.sheets['Summary']
    summary_sheet.set_column('A:A', 30, header_format)
    summary_sheet.set_column('B:B', 40)

print(f"\n✓ Results saved to Excel: {output_filename}")
print(f"  → Sheet 1: 'Results' - All test results with {len(results_df)} rows")
print(f"  → Sheet 2: 'Summary' - Configuration and best results")
print(f"  → The Excel will automatically adapt to any number of iterations you add!")

# 5 - GENERATE COMPARISON PLOTS
print("\nGenerating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: R² vs Iterations
axes[0, 0].plot(results_df['Iterations'], results_df['Test_R2'], 'o-', label='Test R²', color='blue', linewidth=2, markersize=8)
axes[0, 0].plot(results_df['Iterations'], results_df['Train_R2'], 's-', label='Train R²', color='orange', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Iterations', fontsize=12)
axes[0, 0].set_ylabel('R²', fontsize=12)
axes[0, 0].set_title('R² Score vs Iterations', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xscale('log')

# Plot 2: MAE vs Iterations
axes[0, 1].plot(results_df['Iterations'], results_df['Test_MAE'], 'o-', label='Test MAE', color='blue', linewidth=2, markersize=8)
axes[0, 1].plot(results_df['Iterations'], results_df['Train_MAE'], 's-', label='Train MAE', color='orange', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Iterations', fontsize=12)
axes[0, 1].set_ylabel('MAE', fontsize=12)
axes[0, 1].set_title('MAE vs Iterations', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xscale('log')

# Plot 3: MSE vs Iterations
axes[1, 0].plot(results_df['Iterations'], results_df['Test_MSE'], 'o-', label='Test MSE', color='blue', linewidth=2, markersize=8)
axes[1, 0].plot(results_df['Iterations'], results_df['Train_MSE'], 's-', label='Train MSE', color='orange', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Iterations', fontsize=12)
axes[1, 0].set_ylabel('MSE', fontsize=12)
axes[1, 0].set_title('MSE vs Iterations', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xscale('log')

# Plot 4: Computation Time vs Iterations
axes[1, 1].plot(results_df['Iterations'], results_df['Comput_Time_sec']/60, 'o-', color='green', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of Iterations', fontsize=12)
axes[1, 1].set_ylabel('Computation Time (minutes)', fontsize=12)
axes[1, 1].set_title('Computation Time vs Iterations', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xscale('log')

plt.tight_layout()
plot_filename = f'DeepNN_Results_LR{LEARNING_RATE}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"✓ Plots saved to: {plot_filename}")
plt.show()

# 6 - FINAL STATISTICS
print("\n" + "#"*70)
print("### FINAL STATISTICS")
print("#"*70)
best_test_idx = results_df['Test_R2'].idxmax()
best_train_idx = results_df['Train_R2'].idxmax()

print(f"\n🏆 Best Test R²: {results_df.loc[best_test_idx, 'Test_R2']:.6f}")
print(f"   → Achieved at {results_df.loc[best_test_idx, 'Iterations']} iterations")
print(f"   → Test MAE: {results_df.loc[best_test_idx, 'Test_MAE']:.4f}")
print(f"   → Test MSE: {results_df.loc[best_test_idx, 'Test_MSE']:.4f}")
print(f"   → Test RMSE: {results_df.loc[best_test_idx, 'Test_RMSE']:.4f}")

print(f"\n🏆 Best Train R²: {results_df.loc[best_train_idx, 'Train_R2']:.6f}")
print(f"   → Achieved at {results_df.loc[best_train_idx, 'Iterations']} iterations")

print(f"\n⏱️  Total Computation Time: {total_time:.2f}s ({total_time/60:.2f} min)")
print(f"   → Average per test: {total_time/len(ITERATION_TESTS):.2f}s")

print("\n" + "#"*70)
print("### ALL TESTS COMPLETE!")
print("#"*70)
