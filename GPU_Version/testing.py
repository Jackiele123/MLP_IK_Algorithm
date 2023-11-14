import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

import MLP as MLP
import forward_kinematics as FK

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

# Example usage:
# Assuming you have the trained weights (W1, b1, W2, b2) and the mean and std for X and y
new_position = np.array([-7.532041341
, 194.9840623
, 178.6046474
])  # Replace with the actual position values
predicted_angles = MLP.predict_joint_angles(new_position, "weights\weights_epoch_999.txt" , X_mean, X_std, y_mean, y_std)
angles = []
for r in predicted_angles:
    angles.append(r)
    print(r*180/np.pi)
angles.append(0)
print(FK.forward_kinematics(angles)[0])



