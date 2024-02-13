import tensorflow as tf
import numpy as np
import forward_kinematics as FK
import MLP as MLP
import pandas as pd

DATASET = MLP.get_presigned_url("data-1699387684828.csv")

DATA = pd.read_csv(DATASET)

X = DATA[['x', 'y', 'z']].values  # Input features
y = DATA[['j1', 'j2', 'j3', 'j4', 'j5']].values  # Target values
# Calculate the mean and standard deviation for features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

# Calculate the mean and standard deviation for labels
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)

# Load the model
model = tf.keras.models.load_model('my_model.h5', custom_objects={'euclidean_loss': euclidean_loss})

# Function to predict joint angles
def predict_joint_angles(model, position, X_mean, X_std, y_mean, y_std):
    """
    Predicts the joint angles for a given position using the loaded model.

    Parameters:
    - model: The loaded TensorFlow model.
    - position: A numpy array or list of positions (x, y, z) to predict the joint angles for.
    - X_mean, X_std: The mean and standard deviation of the input features used for training.
    - y_mean, y_std: The mean and standard deviation of the target values used for training.

    Returns:
    - predicted_angles: A numpy array of the predicted joint angles.
    """

    # Normalize the input position
    position_normalized = (position - X_mean) / X_std

    # Predict the normalized joint angles
    predicted_normalized_angles = model.predict(position_normalized)

    # Denormalize the predicted joint angles
    predicted_angles = predicted_normalized_angles * y_std + y_mean

    return predicted_angles

# Example usage
position = np.array([[-7.532041341
, 194.9840623
, 178.6046474]])  # Replace with your actual position data
predicted_angles = predict_joint_angles(model, position, X_mean, X_std, y_mean, y_std)

# Print the predicted joint angles
print("Predicted Joint Angles:", predicted_angles)
