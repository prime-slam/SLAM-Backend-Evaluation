import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import open3d as o3d
import plotly.graph_objects as go

def get_normal(points):
  sum = [0, 0, 0]
  for point in points:
    sum += point
  c = sum/len(points)
  print('точки')
  print(points)
  print('точка с')
  print(c)
  A = []
  for point in points:
    A.append(point - c)
  print('a')
  A_array = np.array(A)
  print(A_array)
  U, S, VT = np.linalg.svd(A_array) #вот тут фигня продолжается 
  n = VT[2]
  return n[0], n[1], n[2], np.dot(n, c)


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

extract_data_depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)#unpacking depth
extract_data_colors = cv2.imread('annot.png',cv2.IMREAD_COLOR)#unpacking colors 

color_matrix = np.array(extract_data_colors)#making np array of colors
matrix_depth = np.array(extract_data_depth) #making np array of depth

rows, colums = matrix_depth.shape           #getting the shape of the matrices

matrix_u = np.empty(matrix_depth.shape, dtype=int)
matrix_v = np.empty(matrix_depth.shape, dtype=int)

for i in range (rows):                     #x coordinats
  for j in range (colums):
    matrix_u[i,j] = i

for i in range (rows):                     #y coordinates
  for j in range (colums):
    matrix_v[i,j] = j

reshaped_color_matrix = color_matrix.reshape(-1, color_matrix.shape[2])#reshape matrix in order to get unique colors
colors_unique = np.unique(reshaped_color_matrix, axis=0)

x,y,z, = convert_from_plane_to_3d(matrix_u, matrix_v, matrix_depth, cx = 319.50, cy = 239.50, focal_x = 481.20, focal_y = -480.00)   #getting xyz coordinates of each point

matrix_xyz = np.empty((rows , colums, 3), dtype= float)

for i in range(rows):
  for j in range (colums):
    matrix_xyz[i,j] = (x[i,j], y[i,j], z[i,j])

answ_matrix = matrix_xyz.reshape(rows*colums, 3)#now we have a list of points

print(colors_unique)
num_colors, three = colors_unique.shape

num_of_points, three = reshaped_color_matrix.shape

equations = np.zeros((num_colors, 4))

i = 0
for color in colors_unique:
  if (color !=[0,0,0]).all():        #for every unique color 
    print('цвет')
    print(color)
    plane_points = []
    k = 0
    for k in range(num_of_points):
      v = (reshaped_color_matrix[k] == color).all()
      if (v):
        plane_points.append(answ_matrix[k])
      k += 1
    plane_points_array = np.array(plane_points)
    equations[i] = np.array(get_normal(plane_points_array)) #вот тут начинается фигня)
    print(equations[i])
    print('------------------')
    i += 1
