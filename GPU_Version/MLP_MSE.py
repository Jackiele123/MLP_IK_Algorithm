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

# Define the model
inputs = Input(shape=(3,))
x = Dense(64, activation='relu')(inputs)
outputs = Dense(5, activation='relu')(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model with MSE loss
# model_checkpoint_path = 'model_epoch_68.h5'
# model = load_model(model_checkpoint_path)
model.compile(optimizer='adam', loss='mean_squared_error')


# Callback to save the model every 100 epochs
checkpoint_cb = ModelCheckpoint('model_epoch_{epoch:02d}.h5', save_freq='epoch', save_weights_only=False)

# Assuming 'last_epoch' is the epoch you last completed before saving the checkpoint
last_epoch = 0  # for exampl
new_epochs = 10000  # total epochs you want to train including the previous ones

# Fit the model, specifying the initial epoch
model.fit(
    train_dataset,
    epochs=new_epochs,
    initial_epoch=last_epoch,
    callbacks=[checkpoint_cb]
)
model.save('my_model.h5')

