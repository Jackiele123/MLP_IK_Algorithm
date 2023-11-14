import os
import numpy as np
import pandas as pd
import boto3
from dotenv import load_dotenv
from botocore.client import Config
from scipy.spatial.distance import euclidean

import forward_kinematics as FK

load_dotenv()

# The name of your Wasabi bucket
BUCKET_NAME = 'robotarm'

# The name you want to save the PDF as in Wasabi
FILE_NAME = 'data-1699247538834.csv'

def get_presigned_url(filename):
    """Generate a pre-signed URL for a file in Wasabi."""
    try:
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            aws_access_key_id=os.getenv("WASABI_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("WASABI_SECRET_KEY"),
            endpoint_url='https://s3.ca-central-1.wasabisys.com',
            config=Config(signature_version='s3v4')
        )

        presigned_url = s3.generate_presigned_url('get_object',
                                                  Params={'Bucket': BUCKET_NAME,
                                                          'Key': filename},
                                                  ExpiresIn=3600)  # Link expires in 1 hour
        return presigned_url
    except Exception as e:
        raise e

DATASET = get_presigned_url(FILE_NAME)

DATA = pd.read_csv(DATASET)


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Derivative of tanh for use in backpropagation
def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Euclidean loss function from 3D Distance between two points
def euclidean_loss(X_true_normalized, y_pred_normalized):

    X = DATA[['x', 'y', 'z']].values  # Input features
    y = DATA[['j1', 'j2', 'j3', 'j4', 'j5']].values  # Target values

    # Calculate the mean and standard deviation for features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    # Calculate the mean and standard deviation for labels
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)

    # Denormalize the predicted and true joint angles
    X_true = X_true_normalized * X_std + X_mean
    y_pred = y_pred_normalized * y_std + y_mean
    
    # Initialize list to hold distances
    distances = []
    
    # Iterate over each set of joint angles
    for true_position, pred_angles in zip(X_true, y_pred):
        # Use the FK function to calculate the actual positions from the joint angles
        temp_angles = []
        for angle in pred_angles:
            temp_angles.append(angle)  # Your list of 5 predicted angles
        temp_angles.append(0)  # Append the default sixth angle with value 0
        pred_position = FK.forward_kinematics(temp_angles)[0]
        
        # Calculate the Euclidean distance between the true and predicted positions
        # print(true_position)
        # print(pred_position)
        distance = euclidean(true_position, pred_position)
        distances.append(distance)
    
    # Calculate the mean of the distances
    mean_distance = np.mean(distances)
    
    return mean_distance

# Forward propagation
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = np.tanh(Z2)
    return A1, A2

# Backpropagation
def backprop(X, y, A1, A2, W1, b1, W2, b2, lr=0.01):
    dZ2 = (A2 - y) * (1 - np.tanh(A2) ** 2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (1 - np.tanh(A1) ** 2)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0)

    # Update weights and biases
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

# Training loop
def train(X_train, y_train, W1, b1, W2, b2, epochs=100000, lr=0.01, batch_size=32, save_path='weights'):
    for epoch in range(epochs):
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for batch in range(n_batches):
                # Select the mini-batch
                begin = batch * batch_size
                end = min(begin + batch_size, n_samples)
                X_batch = X_train_shuffled[begin:end]
                y_batch = y_train_shuffled[begin:end]

                # Forward pass
                A1, A2 = forward(X_batch, W1, b1, W2, b2)
                
                # Compute loss
                loss = euclidean_loss(X_batch, A2)

                # Backward pass
                W1, b1, W2, b2 = backprop(X_batch, y_batch, A1, A2, W1, b1, W2, b2, lr)

            # Print the loss at the end of each epoch
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch}, Loss: {loss}')
                # Save the weights and biases to a text file
                with open(f'{save_path}_epoch_{epoch}.txt', 'w') as f:
                    np.savetxt(f, W1, header='W1', comments='')
                    np.savetxt(f, b1, header='b1', comments='')
                    np.savetxt(f, W2, header='W2', comments='')
                    np.savetxt(f, b2, header='b2', comments='')

    return W1, b1, W2, b2

# Test the model
def test(X_test, y_test, W1, b1, W2, b2):
    _, A2_test = forward(X_test, W1, b1, W2, b2)
    loss = euclidean_loss(y_test, A2_test)
    print(f'Test Loss: {loss}')
    return A2_test


def predict_joint_angles(position, file_path, X_mean, X_std, y_mean, y_std):
    """
    Predicts the joint angles for a given position using the trained MLP model.

    Parameters:
    - position: A numpy array of the position (x, y, z) to predict the joint angles for.
    - W1, b1, W2, b2: The trained weights and biases of the MLP model.
    - X_mean, X_std: The mean and standard deviation of the input features used for training.
    - y_mean, y_std: The mean and standard deviation of the target values used for training.

    Returns:
    - predicted_angles: A numpy array of the predicted joint angles.
    """
    with open(file_path, 'r') as file:
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
    W1 = np.array(W1).reshape((3, -1))  # Reshape according to your dimensions, here it's 3 for example
    b1 = np.array(b1)
    W2 = np.array(W2).reshape((-1, 5))  # Reshape according to your dimensions, here it's 5 for example
    b2 = np.array(b2)
    # Normalize the input position
    position_normalized = (position - X_mean) / X_std

    # Perform a forward pass to get the predicted normalized joint angles
    Z1 = np.dot(position_normalized, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    predicted_normalized_angles = np.tanh(Z2)

    # Denormalize the predicted joint angles
    predicted_angles = predicted_normalized_angles * y_std + y_mean

    return predicted_angles





