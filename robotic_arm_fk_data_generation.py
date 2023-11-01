import numpy as np

# DH parameters extracted from the JavaScript file
dh_parameters = [{'d': 87, 'a': 35, 'theta': 0, 'alpha': 1.5707963267948966}, {'d': 0, 'a': 147.25, 'theta': 1.5707963267948966, 'alpha': 0}, {'d': 0, 'a': 55.5, 'theta': -3.141592653589793, 'alpha': 1.5707963267948966}, {'d': 141.6, 'a': 0, 'theta': 0, 'alpha': -1.5707963267948966}, {'d': 0, 'a': 0, 'theta': 0, 'alpha': 1.5707963267948966}, {'d': 80, 'a': 0, 'theta': 0, 'alpha': 0}]
constraints = [{'min': -3.141592653589793/2, 'max': 3.141592653589793}, {'min': -3.141592653589793, 'max': 3.141592653589793}, {'min': -3.141592653589793, 'max': 3.141592653589793}, {'min': -3.141592653589793, 'max': 3.141592653589793}, {'min': -3.141592653589793, 'max': 3.141592653589793}, {'min': -3.141592653589793, 'max': 3.141592653589793}]
def compute_transformation_matrix(theta, d, a, alpha):
    """Compute the transformation matrix using the DH parameters."""
    T = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return T

def forward_kinematics(joint_angles, dh_parameters):
    """Compute the end effector position using forward kinematics."""
    T = np.eye(4)
    for i, angles in enumerate(joint_angles):
        theta = angles + dh_parameters[i]['theta']
        d = dh_parameters[i]['d']
        a = dh_parameters[i]['a']
        alpha = dh_parameters[i]['alpha']
        
        T_i = compute_transformation_matrix(theta, d, a, alpha)
        T = np.dot(T, T_i)
        
    return T[:3, 3]

def generate_training_data(num_samples, dh_parameters):
    """Generate training data using the forward kinematics function."""
    joint_angles_data = []
    end_effector_positions = []
    
    for _ in range(num_samples):
        joint_angles = [np.random.rand() * 2 * np.pi - np.pi for _ in range(6)]
        position = forward_kinematics(joint_angles, dh_parameters)
        
        joint_angles_data.append(joint_angles)
        end_effector_positions.append(position)
    
    return np.array(joint_angles_data), np.array(end_effector_positions)

# Example usage:
joint_angles_data, end_effector_positions_data = generate_training_data(10, dh_parameters)
