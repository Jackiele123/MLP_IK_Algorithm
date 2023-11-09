import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import boto3
import forward_kinematics as FK
from dotenv import load_dotenv
from botocore.client import Config

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

# Read the data
DATA = pd.read_csv(DATASET)



# Continue with your preprocessing...
X = DATA[['x', 'y', 'z']].values
y = DATA[['j1', 'j2', 'j3', 'j4', 'j5']].values

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
X_mean_tensor = tf.constant(X_mean, dtype=tf.float32)
X_std_tensor = tf.constant(X_std, dtype=tf.float32)
y_mean_tensor = tf.constant(y_mean, dtype=tf.float32)
y_std_tensor = tf.constant(y_std, dtype=tf.float32)
X_normalized = (X - X_mean) / X_std
y_normalized = (y - y_mean) / y_std

X_train = X_normalized
y_train = y_normalized

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)  # Adjust buffer_size and batch_size as needed
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Euclidean loss function from 3D Distance between two points
def euclidean_loss(y_true_normalized, y_pred_normalized):
    # Denormalize the predicted and true joint angles
    y_true_denorm = y_true_normalized * y_std_tensor + y_mean_tensor
    y_pred_denorm = y_pred_normalized * y_std_tensor + y_mean_tensor

    # Define a function to compute euclidean distance for a single example
    def compute_distance(true_pred_pair):
        true_angles, pred_angles = true_pred_pair[0], true_pred_pair[1]

        pred_angles = tf.concat([pred_angles, tf.constant([0.0])], axis=0)
        true_angles = tf.concat([true_angles, tf.constant([0.0])], axis=0)
        true_position = FK.forward_kinematics(true_angles)  # This should return a tensor of shape [3]
        pred_position = FK.forward_kinematics(pred_angles)  # This should return a tensor of shape [3]
        
        
        # Compute Euclidean distance using TensorFlow operations
        distance = tf.norm(true_position - pred_position, ord='euclidean')
        
        return distance

    # Use tf.map_fn to apply compute_distance to each pair of true and predicted positions
    distances = tf.map_fn(compute_distance, (y_true_denorm, y_pred_denorm), fn_output_signature=tf.float32)

    # Calculate the mean of the distances
    mean_distance = tf.reduce_mean(distances)
    return mean_distance

# Define the model
inputs = Input(shape=(3,))
x = Dense(64, activation='tanh')(inputs)
outputs = Dense(5, activation='tanh')(x)
model = Model(inputs=inputs, outputs=outputs)

model_checkpoint_path = 'goodEpochs/model_epoch_02.h5'
model = load_model(model_checkpoint_path, custom_objects={'euclidean_loss': euclidean_loss})

model.compile(optimizer='adam', loss=euclidean_loss)

# Callback to save the model every 100 epochs
checkpoint_cb = ModelCheckpoint('model_epoch_{epoch:02d}.h5', save_freq='epoch', save_weights_only=False)

# Assuming 'last_epoch' is the epoch you last completed before saving the checkpoint
last_epoch = 2  # for example
new_epochs = 10000  # total epochs you want to train including the previous ones

# Fit the model, specifying the initial epoch
model.fit(
    train_dataset,
    epochs=new_epochs,
    initial_epoch=last_epoch,
    callbacks=[checkpoint_cb]
)
model.save('my_model.h5')
x v m,b c,mb
# # Predict function
# def predict_joint_angles(position, model):
#     position_normalized = (position - X_mean) / X_std
#     predicted_normalized_angles = model.predict(position_normalized)
#     predicted_angles = predicted_normalized_angles * y_std + y_mean
#     return predicted_angles

# position = np.array([100, 100, 100])  # Example position
# predicted_angles = predict_joint_angles(position, model)
# print(predicted_angles)
