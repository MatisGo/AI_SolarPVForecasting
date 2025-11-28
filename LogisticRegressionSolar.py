import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import openpyxl

# Retrieve data from CSV file (RawData.csv) 

data = pd.read_csv('RawData.csv')
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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler(feature_range=(0,1))
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(0,1))
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = float(0)

    return w, b

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)   # compute activation
    cost = -1/m * np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))   # compute cost
    # print('cost:', cost, 'A:', A, 'Y:', Y)
    dZ = A - Y
    dw = 1/m * np.dot(X,dZ.T)
    db = 1/m * np.sum(dZ)
    grads = {"dw": dw, "db": db} 
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = True):
    costs = []
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        if i == 0:
            cost_first = cost
        # print('w_prev:', w, 'grads:', grads["dw"],"grads shape:", grads["dw"].shape)
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        # print('w_new:', w)
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    # print(costs)
    plt.plot(costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    params = {"w": w, "b": b}
    # print('w_final:', w)
    grads = {"dw": grads["dw"], "db": grads["db"]}
    return params, grads, costs


def predict(w,b,X):
    Y_prediction = sigmoid(np.dot(w.T,X)+b)
    # print(Y_prediction)
    return Y_prediction


def model(train_set_x_normalized, train_set_y_normalized, test_set_x_normalized, test_set_y_normalized, num_iterations = 2000, learning_rate = 0.5, print_cost = True):
    # YOUR CODE STARTS HERE
    w, b = initialize_with_zeros(train_set_x_normalized.shape[0])
#     print('w_ini',w)
#     print(train_set_x_normalized, train_set_y_normalized)
    parameters, grads, costs = optimize(w, b, train_set_x_normalized, train_set_y_normalized, num_iterations, learning_rate, print_cost)
    # optimize returns costs where the first element is the initial cost and the second is the final cost
    # public_tests expects d['costs'] to be a list containing the initial cost as a numpy scalar
    # ensure we return the initial cost as a numpy scalar inside a list
    Y_prediction_test = predict(parameters["w"], parameters["b"], test_set_x_normalized)
    Y_prediction_train = predict(parameters["w"], parameters["b"], train_set_x_normalized)
#   print(Y_prediction_test, Y_prediction_train)
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": parameters["w"],
         "b": parameters["b"],
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d

logistic_regression_model = model(X_train.T, y_train.reshape(1, -1), X_test.T, y_test.reshape(1, -1), num_iterations=10000, learning_rate=0.005, print_cost=True)

# Compare predictions to actual values
Y_predicted = logistic_regression_model["Y_prediction_test"]
y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred = scaler_y.inverse_transform(Y_predicted.reshape(-1, 1)).flatten()

#Show plot of predicted vs actual values for the last few days only
plt.figure(figsize=(10,6))

# Show only the last 200 data points (approximately a few days)
last_n = 100
plt.plot(y_actual[-last_n:], label='Actual', color='orange')
plt.plot(y_pred[-last_n:], label='Predicted', color='blue')
plt.legend()
plt.show()
    
# Evaluate accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prédictions train et test (désnormalisées)
Y_prediction_train = logistic_regression_model["Y_prediction_train"]
y_pred_test = scaler_y.inverse_transform(Y_predicted.reshape(-1, 1)).flatten()
y_pred_train = scaler_y.inverse_transform(Y_prediction_train.reshape(-1, 1)).flatten()
y_actual_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_actual_train = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()

# Métriques Train
mae_train = mean_absolute_error(y_actual_train, y_pred_train)
mse_train = mean_squared_error(y_actual_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_actual_train, y_pred_train)

# Métriques Test
mae_test = mean_absolute_error(y_actual_test, y_pred_test)
mse_test = mean_squared_error(y_actual_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_actual_test, y_pred_test)

print("\n========== OVERFITTING/UNDERFITTING ANALYSIS ==========")
print(f"{'Metric':<10} {'Train':>12} {'Test':>12} {'Diff':>12}")
print("-" * 50)
print(f"{'MAE':<10} {mae_train:>12.4f} {mae_test:>12.4f} {mae_test - mae_train:>12.4f}")
print(f"{'MSE':<10} {mse_train:>12.4f} {mse_test:>12.4f} {mse_test - mse_train:>12.4f}")
print(f"{'RMSE':<10} {rmse_train:>12.4f} {rmse_test:>12.4f} {rmse_test - rmse_train:>12.4f}")
print(f"{'R²':<10} {r2_train:>12.4f} {r2_test:>12.4f} {r2_test - r2_train:>12.4f}")
print("-" * 50)

# Diagnostic
if r2_train > 0.9 and r2_test < 0.7:
    print("⚠️  OVERFITTING: Train R² élevé, Test R² faible")
elif r2_train < 0.5 and r2_test < 0.5:
    print("⚠️  UNDERFITTING: Performances faibles sur Train et Test")
elif abs(r2_train - r2_test) < 0.1:
    print("✅ BON FIT: Performances similaires Train/Test")
else:
    print("⚠️  POSSIBLE OVERFITTING: Écart notable entre Train et Test")