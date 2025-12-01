import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dnn_app_utils_extended import *

# ========================================================================
# HYPERPARAMETERS - EASY TO CHANGE
# ========================================================================
LEARNING_RATE = 0.1                    # Learning rate for gradient descent
ITERATION_TESTS = [ 500,1000,2000,5000,10000,25000, 50000, 100000]  # Different iteration counts to test
LAYERS_DIMS = [7, 10, 10, 10, 10, 1]   # Network architecture
PRINT_COST = False                      # Print cost during training
TEST_SIZE = 0.2                         # Train/test split ratio
RANDOM_STATE = 42                       # Random seed for reproducibility

# Activation functions to test on the LAST LAYER
ACTIVATION_FUNCTIONS = ['sigmoid', 'linear', 'relu', 'tanh']
# ========================================================================

print("="*70)
print("TESTING DIFFERENT ACTIVATION FUNCTIONS FOR THE LAST LAYER")
print("="*70)
print(f"Activation functions to test: {ACTIVATION_FUNCTIONS}")
print(f"Iteration counts to test: {ITERATION_TESTS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Architecture: {LAYERS_DIMS}")
print("="*70)

# 1 - LOAD AND PREPROCESS DATA
print("\n" + "="*70)
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

# 2 - DEFINE L-LAYER MODEL FUNCTION WITH CONFIGURABLE LAST ACTIVATION
def L_layer_model(X, Y, layers_dims, learning_rate=0.85, num_iterations=3000,
                  print_cost=False, last_activation="sigmoid"):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->LAST_ACTIVATION.

    Arguments:
    X -- data, numpy array of shape (num_features, number of examples)
    Y -- true label vector, shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    last_activation -- activation function for the last layer: "sigmoid", "linear", "relu", or "tanh"

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

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> LAST_ACTIVATION
        AL, caches = L_model_forward(X, parameters, last_activation=last_activation)

        # Compute cost
        cost = compute_cost(AL, Y, last_activation=last_activation)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches, last_activation=last_activation)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {np.squeeze(cost)}")
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

# 3 - RUN TESTS FOR DIFFERENT ACTIVATION FUNCTIONS AND ITERATIONS
print("\n" + "="*70)
print("RUNNING TESTS FOR DIFFERENT ACTIVATION FUNCTIONS AND ITERATIONS")
print("="*70)

# Store results for all tests
all_results = []
total_start_time = time.time()

test_num = 0
total_tests = len(ACTIVATION_FUNCTIONS) * len(ITERATION_TESTS)

for activation_fn in ACTIVATION_FUNCTIONS:
    for num_iter in ITERATION_TESTS:
        test_num += 1
        print(f"\n{'#'*70}")
        print(f"### TEST {test_num}/{total_tests}: {activation_fn.upper()} - {num_iter} ITERATIONS")
        print(f"{'#'*70}")

        # Start timing for this test
        start_time = time.time()

        # Train the model
        print(f"Training model with {activation_fn} activation and {num_iter} iterations...")
        parameters, costs = L_layer_model(X_train, y_train, LAYERS_DIMS,
                                          learning_rate=LEARNING_RATE,
                                          num_iterations=num_iter,
                                          print_cost=PRINT_COST,
                                          last_activation=activation_fn)

        # End timing
        training_time = time.time() - start_time

        print(f"✓ Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")

        # Make predictions
        pred_train = predict(X_train, y_train, parameters, last_activation=activation_fn)
        pred_test = predict(X_test, y_test, parameters, last_activation=activation_fn)

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
            'Activation': activation_fn.upper(),
            'Iterations': num_iter,
            'Test_R2': r2_test,
            'Test_MAE': mae_test,
            'Test_MSE': mse_test,
            'Test_RMSE': rmse_test,
            'Train_R2': r2_train,
            'Train_MAE': mae_train,
            'Train_MSE': mse_train,
            'Train_RMSE': rmse_train,
            'Comput_Time_sec': training_time,
            'Final_Cost': costs[-1]
        })

        # PRINT DETAILED RESUME FOR THIS TEST
        print("\n" + "="*70)
        print(f"RESUME - TEST {test_num}: {activation_fn.upper()} - {num_iter} ITERATIONS")
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
        print(f"Final Cost: {costs[-1]:.6f}")
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
results_df = results_df[['Activation', 'Iterations',
                         'Test_R2', 'Test_MAE', 'Test_MSE', 'Test_RMSE',
                         'Train_R2', 'Train_MAE', 'Train_MSE', 'Train_RMSE',
                         'Comput_Time_sec', 'Final_Cost']]

# Display the complete results table in console
print("\n" + results_df.to_string(index=False))

# Save to Excel with formatting
iter_str = "_".join(map(str, ITERATION_TESTS))
output_filename = f'Activation_Functions_Comparison_LR{LEARNING_RATE}_Iter{iter_str}.xlsx'

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
    worksheet.set_column('A:A', 12)                    # Activation
    worksheet.set_column('B:B', 12, integer_format)    # Iterations
    worksheet.set_column('C:C', 12, number_format)     # Test_R2
    worksheet.set_column('D:D', 12, number_format)     # Test_MAE
    worksheet.set_column('E:E', 12, number_format)     # Test_MSE
    worksheet.set_column('F:F', 12, number_format)     # Test_RMSE
    worksheet.set_column('G:G', 12, number_format)     # Train_R2
    worksheet.set_column('H:H', 12, number_format)     # Train_MAE
    worksheet.set_column('I:I', 12, number_format)     # Train_MSE
    worksheet.set_column('J:J', 12, number_format)     # Train_RMSE
    worksheet.set_column('K:K', 16, time_format)       # Comput_Time_sec
    worksheet.set_column('L:L', 12, number_format)     # Final_Cost

    # Write headers with formatting
    for col_num, value in enumerate(results_df.columns.values):
        worksheet.write(0, col_num, value, header_format)

    # Add a summary sheet
    best_test_r2_idx = results_df['Test_R2'].idxmax()
    best_train_r2_idx = results_df['Train_R2'].idxmax()

    summary_df = pd.DataFrame({
        'Parameter': ['Learning Rate', 'Architecture', 'Test Size', 'Random State',
                      'Iteration Tests', 'Total Tests Run', 'Total Computation Time (sec)', 'Total Computation Time (min)',
                      'Best Test R²', 'Best Config (Test R²)',
                      'Best Train R²', 'Best Config (Train R²)',
                      'Fastest Training', 'Fastest Config'],
        'Value': [LEARNING_RATE, str(LAYERS_DIMS), TEST_SIZE, RANDOM_STATE,
                  str(ITERATION_TESTS), len(results_df), f'{total_time:.2f}', f'{total_time/60:.2f}',
                  f'{results_df.loc[best_test_r2_idx, "Test_R2"]:.6f}',
                  f'{results_df.loc[best_test_r2_idx, "Activation"]} - {results_df.loc[best_test_r2_idx, "Iterations"]} iter',
                  f'{results_df.loc[best_train_r2_idx, "Train_R2"]:.6f}',
                  f'{results_df.loc[best_train_r2_idx, "Activation"]} - {results_df.loc[best_train_r2_idx, "Iterations"]} iter',
                  f'{results_df["Comput_Time_sec"].min():.2f}s',
                  f'{results_df.loc[results_df["Comput_Time_sec"].idxmin(), "Activation"]} - {results_df.loc[results_df["Comput_Time_sec"].idxmin(), "Iterations"]} iter']
    })

    summary_df.to_excel(writer, sheet_name='Summary', index=False)

    # Format summary sheet
    summary_sheet = writer.sheets['Summary']
    summary_sheet.set_column('A:A', 30, header_format)
    summary_sheet.set_column('B:B', 40)

print(f"\n✓ Results saved to Excel: {output_filename}")
print(f"  → Sheet 1: 'Results' - Comparison of all {len(results_df)} tests")
print(f"  → Sheet 2: 'Summary' - Configuration and best results")

# 5 - GENERATE COMPARISON PLOTS
print("\nGenerating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Create labels for x-axis (Activation + Iterations)
labels = [f"{row['Activation']}\n{row['Iterations']}" for _, row in results_df.iterrows()]
x_pos = np.arange(len(labels))

# Plot 1: R² Comparison
axes[0, 0].bar(x_pos - 0.2, results_df['Test_R2'], 0.4, label='Test R²', color='blue', alpha=0.7)
axes[0, 0].bar(x_pos + 0.2, results_df['Train_R2'], 0.4, label='Train R²', color='orange', alpha=0.7)
axes[0, 0].set_xlabel('Activation Function + Iterations', fontsize=12)
axes[0, 0].set_ylabel('R²', fontsize=12)
axes[0, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: MAE Comparison
axes[0, 1].bar(x_pos - 0.2, results_df['Test_MAE'], 0.4, label='Test MAE', color='blue', alpha=0.7)
axes[0, 1].bar(x_pos + 0.2, results_df['Train_MAE'], 0.4, label='Train MAE', color='orange', alpha=0.7)
axes[0, 1].set_xlabel('Activation Function + Iterations', fontsize=12)
axes[0, 1].set_ylabel('MAE', fontsize=12)
axes[0, 1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: RMSE Comparison
axes[1, 0].bar(x_pos - 0.2, results_df['Test_RMSE'], 0.4, label='Test RMSE', color='blue', alpha=0.7)
axes[1, 0].bar(x_pos + 0.2, results_df['Train_RMSE'], 0.4, label='Train RMSE', color='orange', alpha=0.7)
axes[1, 0].set_xlabel('Activation Function + Iterations', fontsize=12)
axes[1, 0].set_ylabel('RMSE', fontsize=12)
axes[1, 0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Computation Time Comparison
bars = axes[1, 1].bar(x_pos, results_df['Comput_Time_sec']/60, color='green', alpha=0.7)
axes[1, 1].set_xlabel('Activation Function + Iterations', fontsize=12)
axes[1, 1].set_ylabel('Computation Time (minutes)', fontsize=12)
axes[1, 1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_filename = f'Activation_Functions_Comparison_LR{LEARNING_RATE}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"✓ Plots saved to: {plot_filename}")
plt.show()

# 6 - FINAL STATISTICS
print("\n" + "#"*70)
print("### FINAL STATISTICS")
print("#"*70)
best_test_idx = results_df['Test_R2'].idxmax()
best_train_idx = results_df['Train_R2'].idxmax()
fastest_idx = results_df['Comput_Time_sec'].idxmin()

print(f"\n🏆 Best Test R²: {results_df.loc[best_test_idx, 'Test_R2']:.6f}")
print(f"   → Configuration: {results_df.loc[best_test_idx, 'Activation']} with {results_df.loc[best_test_idx, 'Iterations']} iterations")
print(f"   → Test MAE: {results_df.loc[best_test_idx, 'Test_MAE']:.4f}")
print(f"   → Test MSE: {results_df.loc[best_test_idx, 'Test_MSE']:.4f}")
print(f"   → Test RMSE: {results_df.loc[best_test_idx, 'Test_RMSE']:.4f}")

print(f"\n🏆 Best Train R²: {results_df.loc[best_train_idx, 'Train_R2']:.6f}")
print(f"   → Configuration: {results_df.loc[best_train_idx, 'Activation']} with {results_df.loc[best_train_idx, 'Iterations']} iterations")

print(f"\n⚡ Fastest Training: {results_df.loc[fastest_idx, 'Comput_Time_sec']:.2f}s")
print(f"   → Configuration: {results_df.loc[fastest_idx, 'Activation']} with {results_df.loc[fastest_idx, 'Iterations']} iterations")

print(f"\n⏱️  Total Computation Time: {total_time:.2f}s ({total_time/60:.2f} min)")
print(f"   → Average per test: {total_time/total_tests:.2f}s")
print(f"   → Total tests run: {total_tests} ({len(ACTIVATION_FUNCTIONS)} activations × {len(ITERATION_TESTS)} iteration counts)")

print("\n" + "#"*70)
print("### ALL TESTS COMPLETE!")
print("#"*70)

# 7 - RECOMMENDATIONS
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print(f"For regression tasks like solar PV forecasting:")
print(f"  1. Best overall: {results_df.loc[best_test_idx, 'Activation']} with {results_df.loc[best_test_idx, 'Iterations']} iterations")
print(f"     (Test R² = {results_df.loc[best_test_idx, 'Test_R2']:.6f})")
print(f"  2. Fastest: {results_df.loc[fastest_idx, 'Activation']} with {results_df.loc[fastest_idx, 'Iterations']} iterations")
print(f"     (Time = {results_df.loc[fastest_idx, 'Comput_Time_sec']:.2f}s)")
print(f"\nNote: LINEAR activation is typically recommended for regression problems")
print(f"      as it doesn't constrain the output range.")
print("="*70)
