import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
from botocore.client import Config
import boto3
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

# Define the model
inputs = Input(shape=(3,))
x = Dense(64, activation='tanh')(inputs)
outputs = Dense(5, activation='tanh')(x)
model = Model(inputs=inputs, outputs=outputs)






