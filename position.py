import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
# Increase the default font size
plt.rcParams.update({'font.size': 16})  # You can adjust the value '14' to your preference
# Function to plot convex hull for a given set of points
def plot_convex_hull(ax, points, color='k'):
    if len(points) < 3:
        # A convex hull is not well-defined for less than 3 points
        return

    # Computing the convex hull
    hull = ConvexHull(points)

    # Extracting the vertices for the convex hull
    hull_vertices = hull.vertices
    hull_vertices = np.append(hull_vertices, hull.vertices[0]) # Loop back to the first point

    # Plotting the convex hull
    ax.plot(points[hull_vertices, 0], points[hull_vertices, 1], color=color, linestyle='--', lw=2)

# Load the data
file_path = 'output_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Calculate the third quartile (Q3) for the Distance column
q3_distance = data['Distance'].quantile(0.75)

# Filter the data to include only rows where the distance is less than the third quartile
data_below_third_quartile = data[data['Distance'] < q3_distance]

# Creating 2D plane plots with convex hulls for the data below the third quartile
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# XY plane
xy_operating_points = data[['Target_X', 'Target_Y']].values
xy_points_below_third_quartile = data_below_third_quartile[['Target_X', 'Target_Y']].values
axes[1].scatter(xy_points_below_third_quartile[:, 0], xy_points_below_third_quartile[:, 1], color='r')
plot_convex_hull(axes[1], xy_operating_points)
axes[1].set_xlabel('Target X [mm]')
axes[1].set_ylabel('Target Y [mm]')
axes[1].set_title('XY Plane with Loss < Third Quartile')

# XZ plane
xz_operating_points = data[['Target_X', 'Target_Z']].values
xz_points_below_third_quartile = data_below_third_quartile[['Target_X', 'Target_Z']].values
axes[0].scatter(xz_points_below_third_quartile[:, 0], xz_points_below_third_quartile[:, 1], color='g')
plot_convex_hull(axes[0], xz_operating_points)
axes[0].set_xlabel('Target X [mm]')
axes[0].set_ylabel('Target Z [mm]')
axes[0].set_title('XZ Plane with Loss < Third Quartile')

# YZ plane
yz_operating_points = data[['Target_Z', 'Target_Y']].values
yz_points_below_third_quartile = data_below_third_quartile[['Target_Z', 'Target_Y']].values
axes[2].scatter(yz_points_below_third_quartile[:, 0], yz_points_below_third_quartile[:, 1], color='b')
plot_convex_hull(axes[2], yz_operating_points)
axes[2].set_xlabel('Target Z [mm]')
axes[2].set_ylabel('Target Y [mm]')
axes[2].set_title('YZ Plane with Loss < Third Quartile')

plt.tight_layout()
plt.show()