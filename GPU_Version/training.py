import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

import MLP as MLP

DATASET = MLP.get_presigned_url("data-1699387684828.csv")

DATA = pd.read_csv(DATASET)


X = DATA[['x', 'y', 'z']].values  # Input features
y = DATA[['j1', 'j2', 'j3', 'j4', 'j5']].values  # Target values

# Normalize the data if necessary
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = (y - y.mean(axis=0)) / y.std(axis=0)

# Prepare the training and test data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the MLP architecture
input_size = 3  # Number of input features
hidden_size = 20  # Number of neurons in the hidden layer
output_size = 5  # Number of output values


with open('weights\weights_epoch_260.txt', 'r') as file:
    weights_biases_text = file.read()
    lines = weights_biases_text.strip().split('\n')
    W1, b1, W2, b2 = [], [], [], []
    current_matrix = None

    for line in lines:
        if 'W1' in line:
            current_matrix = W1
        elif 'b1' in line:
            current_matrix = b1
        elif 'W2' in line:
            current_matrix = W2
        elif 'b2' in line:
            current_matrix = b2
        else:
            current_matrix.extend(map(float, line.split()))

    # Convert lists to numpy arrays
    W1 = np.array(W1).reshape((3, 20))  # Reshape according to your dimensions, here it's 3 for example
    b1 = np.array(b1)
    W2 = np.array(W2).reshape((20, 5))  # Reshape according to your dimensions, here it's 5 for example
    b2 = np.array(b2)

W1, b1, W2, b2 = MLP.train(X_train, y_train, W1, b1, W2, b2)

