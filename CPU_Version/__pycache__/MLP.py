import os
import sys
import autograd.numpy as np
from autograd import grad
import pandas as pd
import boto3
from dotenv import load_dotenv
from botocore.client import Config
import time

import forward_kinematics as FK

load_dotenv()

# The name of your Wasabi bucket
BUCKET_NAME = 'robotarm'

# The name you want to save the PDF as in Wasabi
FILE_NAME = 'data-1699387684828.csv'

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


X = DATA[['x', 'y', 'z']].values  # Input features
y = DATA[['j1', 'j2', 'j3', 'j4', 'j5']].values  # Target values

def tanh(x):
    return np.tanh(x)
# Derivative of tanh for use in backpropagation
def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Euclidean loss function from 3D Distance between two points
def euclidean_loss(X_true, y_pred):
    # Calculate the predicted positions for each pair of joint angles
    y_pred_positions = np.array([FK.forward_kinematics(pred_angles) for pred_angles in y_pred])
    # Calculate the Euclidean distances for each pair of points
    distances = np.sqrt(np.sum((X_true - y_pred_positions) ** 2, axis=1))
    # Calculate the mean of the distances
    mean_distance = np.mean(distances)
    return mean_distance
# Function from autograd library to compute the gradient of the loss function
euclidean_loss_grad = grad(euclidean_loss, 1)

# Forward propagation
def forward(X, W1, b1, W2, b2):
    # W1 has shape (3, 20), b1 has shape (20,)
    # W2 has shape (20, 5), b2 has shape (5,)
    Z1 = np.dot(X, W1) + b1 # Z1 has shape (32, 20)
    A1 = tanh(Z1) # A1 has shape (32, 20)
    Z2 = np.dot(A1, W2) + b2 # Z2 has shape (32, 5)
    A2 = tanh(Z2) # A2 has shape (32, 5)
    return A1, A2

# Backpropagation
def backprop(X, A1, A2, W1, b1, W2, b2, lr=0.01):
# X = y_i (0) || A1 = y_i^(1) || A2 = y_i^(2) || L is the loss function
    # This is ∂L/∂A2 
    grad_loss_joint = euclidean_loss_grad(X, A2)
    # This is ∂L/∂Z2 = ∂L/∂A2 * ∂A2/∂Z2
    dZ2 = grad_loss_joint * tanh_derivative(A2)
    # This is ∂L/∂W2 = ∂L/∂A2 * ∂A2/∂Z2 * ∂Z2/∂W2 || A1.T is ∂Z2/∂W2
    dW2 = np.dot(A1.T, dZ2)
    # This is ∂L/∂A2 * ∂A2/∂Z2 
    db2 = np.sum(dZ2, axis=0)
    # This is ∂L/∂A1 = ∂L/∂Z2 * ∂Z2/∂A1 || W2.T is ∂Z2/∂A1
    dA1 = np.dot(dZ2, W2.T)
    # This is ∂L/∂Z1 = ∂L/∂A1 * ∂A1/∂Z1
    dZ1 = dA1 * tanh_derivative(A1)
    # This is ∂L/∂W1 = ∂L/∂Z1 * ∂Z1/∂W1 || X.T is ∂Z1/∂W1
    dW1 = np.dot(X.T, dZ1)
    # This is ∂L/∂b1 = ∂L/∂Z1 = sum of the changes in the Z1 values
    db1 = np.sum(dZ1, axis=0)
# Update weights and biases
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

# Training loop
def train(X_train, y_train, W1, b1, W2, b2, epochs=10000, lr=0.01, batch_size=32, save_path='weights'):
    for epoch in range(epochs):
        last_print_time = time.time()  # Initialize the last print time
        start_time = time.time()  # Initialize the start time
        batch_times = []
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        batch_times = []
        for batch in range(n_batches):
            batch_start_time = time.time()
            # Select the mini-batch
            begin = batch * batch_size
            end = min(begin + batch_size, n_samples)
            X_batch = X_train[begin:end]

            # Forward pass
            A1, A2 = forward(X_batch, W1, b1, W2, b2)

            # Backward pass
            W1, b1, W2, b2 = backprop(X_batch, A1, A2, W1, b1, W2, b2, lr)
            # Calculate and store the time taken for this batch
            end_time = time.time()
            batch_times.append(end_time - batch_start_time)

            # Calculate the average time per batch
            avg_batch_time = np.mean(batch_times)

            # Estimate time for the epoch
            estimated_time = avg_batch_time * n_batches
            current_time = time.time()
            loss = euclidean_loss(X_batch, A2)
            if current_time - last_print_time >= 1:  # Check if 1 second have passed
                sys.stdout.write(f'\rEpoch {epoch}/{epochs}, {batch + 1}/{n_batches}: {np.sum(batch_times):.2f}s, Estimated Time = {estimated_time:.2f}s, Loss: {loss}')
                sys.stdout.flush()
                last_print_time = current_time
        # Print the loss at the end of each epoch
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f'Epoch {epoch} completed in {epoch_duration:.2f} seconds')
        print(f'Epoch {epoch}, Loss: {loss}')
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Save the weights and biases to a text file
            with open(f'{save_path}_epoch_{epoch}.txt', 'w') as f:
                np.savetxt(f, W1, header='W1', comments='')
                np.savetxt(f, b1, header='b1', comments='')
                np.savetxt(f, W2, header='W2', comments='')
                np.savetxt(f, b2, header='b2', comments='')

    return W1, b1, W2, b2

# Test the model
def test(X_test, y_test, W1, b1, W2, b2):
    A2_test = forward(X_test, W1, b1, W2, b2)
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



def euclidean_dist(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1: A numpy array representing the first point (x, y, z).
    - point2: A numpy array representing the second point (x, y, z).

    Returns:
    - distance: The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))




