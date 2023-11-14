import os
import numpy as np
import pandas as pd
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
# from tensorflow import keras
# import boto3
# import forward_kinematics as FK
# from dotenv import load_dotenv
# from botocore.client import Config
# from scipy.spatial.distance import euclidean
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    