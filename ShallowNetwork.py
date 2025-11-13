import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Retrieve data from CSV file (RawData.csv) 

data = pd.read_csv('Project/RawData.csv')
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


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    #(≈ 3 lines of code)

    n_x = X.shape[0]  # size of input layer, X.shape == (n_features, n_examples)
    n_h = 4           # we choose the size of the hidden layer to be
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y):
    """
    Computes the cost given in equation (10)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cost given equation (10)
    
    """
    
    m = Y.shape[1] # number of examples
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    
    cost = float(np.squeeze(cost))
                                    
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    W1 = copy.deepcopy(parameters["W1"])
    b1 = parameters["b1"]
    W2 = copy.deepcopy(parameters["W2"])
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 1000, learning_rate=0.85, print_cost=True):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        
        parameters = update_parameters(parameters, grads)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model
    """
    
    A2, cache = forward_propagation(X, parameters)
    predictions = A2
    
    
    return predictions

# run model 

X_train_T = X_train.T
y_train_reshaped = y_train.reshape(1, -1)
X_test_T = X_test.T
y_test_reshaped = y_test.reshape(1, -1)

learning_rate = .85
num_iterations = 2000
hidden_layer_sizes = [2, 4, 8, 16, 32, 64]
for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_model(X_train_T, y_train_reshaped, n_h, print_cost=False)
    Y_prediction_train = predict(parameters, X_train_T)
    Train_accuracy=(100 - np.mean(np.abs(Y_prediction_train.flatten() - y_train)) * 100)
    
    Y_prediction_test = predict(parameters, X_test_T)
    Test_accuracy=(100 - np.mean(np.abs(Y_prediction_test.flatten() - y_test)) * 100)
    
        
    print ("Train Accuracy for {} hidden units: {} %".format(n_h, Train_accuracy))
    print ("Test Accuracy for {} hidden units: {} %".format(n_h, Test_accuracy))
    
    # --- Evaluation metrics (inverse-transform predictions) ---
    try:
        y_pred_inv_metrics = scaler_y.inverse_transform(Y_prediction_test.flatten().reshape(-1, 1)).flatten()
        y_actual_inv_metrics = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(y_actual_inv_metrics, y_pred_inv_metrics)
        mse = mean_squared_error(y_actual_inv_metrics, y_pred_inv_metrics)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual_inv_metrics, y_pred_inv_metrics)

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
    except Exception as e:
        print('Could not compute evaluation metrics:', e)
    
    # --- Plot comparison: actual vs predicted for a few days (first ~72 points) ---
    try:
        # Inverse transform predictions and ground truth back to original scale
        y_pred_test_inv = scaler_y.inverse_transform(Y_prediction_test.flatten().reshape(-1, 1)).flatten()
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Number of points to show (approx. 3 days if hourly data)
        n_points = min(72, len(y_test_inv))

        plt.figure(figsize=(12, 5))
        plt.plot(range(n_points), y_test_inv[:n_points], label='Actual')
        plt.plot(range(n_points), y_pred_test_inv[:n_points], label='Predicted')
        plt.title(f'Actual vs Predicted (first {n_points} test samples)')
        plt.xlabel('Sample index (test set)')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Show the plot. In interactive environments this will pop up; otherwise it will be saved.
        # try:
        #     plt.show()
        # except Exception:
        #     # If show() is not available (headless), save to a file instead
        #     out_path = f'Project/prediction_vs_actual_nh_{n_h}.png'
        #     plt.savefig(out_path)
        #     print(f'Plot saved to {out_path}')
    except Exception as e:
        print('Could not create comparison plot:', e)
    
