import tensorflow as tf
import numpy as np

# Define the DH parameters as constants in TensorFlow
DHParameters = [
    # Base to Joint 1
    {'d': tf.constant(87, dtype=tf.float32), 'a': tf.constant(35, dtype=tf.float32), 'theta': tf.constant(0, dtype=tf.float32), 'alpha': tf.constant(np.pi / 2, dtype=tf.float32)},
    # Joint 1 to Joint 2
    {'d': tf.constant(0, dtype=tf.float32), 'a': tf.constant(147.25, dtype=tf.float32), 'theta': tf.constant(np.pi / 2, dtype=tf.float32), 'alpha': tf.constant(0, dtype=tf.float32)},
    # Joint 2 to Joint 3
    {'d': tf.constant(0, dtype=tf.float32), 'a': tf.constant(55.5, dtype=tf.float32), 'theta': tf.constant(-np.pi, dtype=tf.float32), 'alpha': tf.constant(np.pi / 2, dtype=tf.float32)},
    # Joint 3 to Joint 4
    {'d': tf.constant(141.6, dtype=tf.float32), 'a': tf.constant(0, dtype=tf.float32), 'theta': tf.constant(0, dtype=tf.float32), 'alpha': tf.constant(-np.pi / 2, dtype=tf.float32)},
    # Joint 4 to Joint 5
    {'d': tf.constant(0, dtype=tf.float32), 'a': tf.constant(0, dtype=tf.float32), 'theta': tf.constant(0, dtype=tf.float32), 'alpha': tf.constant(np.pi / 2, dtype=tf.float32)},
    # Joint 5 to End Effector
    {'d': tf.constant(80, dtype=tf.float32), 'a': tf.constant(0, dtype=tf.float32), 'theta': tf.constant(0, dtype=tf.float32), 'alpha': tf.constant(0, dtype=tf.float32)},
]

def get_transformation_matrix(d, a, theta, alpha):
    # Ensure that all inputs are of type float32
    d = tf.cast(d, tf.float32)
    a = tf.cast(a, tf.float32)
    theta = tf.cast(theta, tf.float32)
    alpha = tf.cast(alpha, tf.float32)

    ct = tf.cos(theta)
    st = tf.sin(theta)
    ca = tf.cos(alpha)
    sa = tf.sin(alpha)
    
    return tf.convert_to_tensor([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ], dtype=tf.float32)


def forward_kinematics(joint_angles):
    # Ensure joint_angles is a tensor, append a zero for the 6th angle
    # joint_angles = tf.concat([joint_angles, tf.constant([0.0], dtype=tf.float32)], 0)
    
    transform_matrix = tf.eye(4, dtype=tf.float32)
    
    for i, params in enumerate(DHParameters):
        # Use the modified joint_angles with the appended zero for the 6th angle
        theta = joint_angles[i] + params['theta']
        next_transform_matrix = get_transformation_matrix(params['d'], params['a'], theta, params['alpha'])
        transform_matrix = tf.linalg.matmul(transform_matrix, next_transform_matrix)
    
    # Extract the position from the final transformation matrix
    position = transform_matrix[:3, 3]
    position = tf.stack([position[0], position[2], -position[1]])
    
    return position  # Return only position for now