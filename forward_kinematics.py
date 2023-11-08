import numpy as np
from scipy.spatial.transform import Rotation as R

# Define the DH parameters as given in the JavaScript code
DHParameters = [
    # Base to Joint 1
    {'d': 87, 'a': 35, 'theta': 0, 'alpha': np.pi / 2},
    # Joint 1 to Joint 2
    {'d': 0, 'a': 147.25, 'theta': np.pi / 2, 'alpha': 0},
    # Joint 2 to Joint 3
    {'d': 0, 'a': 55.5, 'theta': -np.pi, 'alpha': np.pi / 2},
    # Joint 3 to Joint 4
    {'d': 141.6, 'a': 0, 'theta': 0, 'alpha': -np.pi / 2},
    # Joint 4 to Joint 5
    {'d': 0, 'a': 0, 'theta': 0, 'alpha': np.pi / 2},
    # Joint 5 to End Effector
    {'d': 80, 'a': 0, 'theta': 0, 'alpha': 0},
]

def get_transformation_matrix(d, a, theta, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(joint_angles):
    transform_matrix = np.identity(4)
    
    for i, params in enumerate(DHParameters):
        theta = joint_angles[i] + params['theta']
        next_transform_matrix = get_transformation_matrix(params['d'], params['a'], theta, params['alpha'])
        transform_matrix = np.dot(transform_matrix, next_transform_matrix)
    
    # Extract the position from the final transformation matrix
    # Swapping the Y and Z axes
    position = transform_matrix[:3, 3]
    position[1], position[2] = position[2], -position[1]
    
    # Extract the orientation from the final transformation matrix
    orientation = R.from_matrix(transform_matrix[:3, :3]).as_euler('xyz', degrees=True)
    
    return position, orientation