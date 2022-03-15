import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import open3d as o3d
import plotly.graph_objects as go

def convert_from_plane_to_3d(u, v, depth, cx, cy, focal_x, focal_y):
  matrix_cx = np.empty(u.shape, dtype= int)
  matrix_cy = np.empty(u.shape, dtype= int)

  for i in range (rows):
    for j in range (colums):
      matrix_cx[i,j] = cx

  for i in range (rows):
    for j in range (colums):
      matrix_cy[i,j] = cy

  x_over_z = (matrix_cx - u) * 1/focal_x
  y_over_z = (matrix_cy - v)  * 1/focal_y

  z_matrix = np.empty(u.shape)

  for i in range (rows):
    for j in range(colums):
      z_matrix[i,j] = depth[i,j]/ 5000 

  x_matrix = x_over_z * z_matrix
  y_matrix = y_over_z * z_matrix
  return x_matrix, y_matrix, z_matrix

extract_data_depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)
extract_data_colors = cv2.imread('annot.png',cv2.IMREAD_COLOR)


color_matrix = np.array(extract_data_colors)
rows, colums, three = color_matrix.shape
color_matrix_right_shape = color_matrix.reshape(rows* colums, 3)

matrix_depth = np.array(extract_data_depth)
rows, colums = matrix_depth.shape

matrix_u = np.empty(matrix_depth.shape, dtype=int)
matrix_v = np.empty(matrix_depth.shape, dtype=int)

for i in range (rows):
  for j in range (colums):
    matrix_u[i,j] = i

for i in range (rows):
  for j in range (colums):
    matrix_v[i,j] = j

x,y,z, = convert_from_plane_to_3d(matrix_u, matrix_v, matrix_depth, cx = 319.50, cy = 239.50, focal_x = 481.20, focal_y = -480.00)

matrix_xyz = np.empty((rows , colums, 3), dtype= float)

for i in range(rows):
  for j in range (colums):
    matrix_xyz[i,j] = (x[i,j], y[i,j], z[i,j])

answ_matrix = matrix_xyz.reshape(rows*colums, 3)

pc = o3d.geometry.PointCloud()


pc.points = o3d.utility.Vector3dVector(answ_matrix)
points = np.asarray(pc.points)
pc.colors = o3d.utility.Vector3dVector(color_matrix_right_shape)

colors = None
if pc.has_colors():
    colors = np.asarray(pc.colors)
elif pc.has_normals():
    colors = (0.5, 0.5, 0.5) + np.asarray(pc.normals) * 0.5
else:
    pc.paint_uniform_color((1.0, 0.0, 0.0))
    colors = np.asarray(pc.colors)

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=points[:,0], y=points[:,1], z=points[:,2], 
            mode='markers',
            marker=dict(size=1, color=colors)
        )
    ],
    layout=dict(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
)
fig.show()
