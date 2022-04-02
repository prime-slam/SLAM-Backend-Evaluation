import numpy as np
import math
import cv2
import open3d as o3d


def convert_from_plane_to_3d(u, v, depth, cx, cy, focal_x, focal_y):
    x_over_z = (cx - u) * 1 / focal_x
    y_over_z = (cy - v) * 1 / focal_y

    z_matrix = depth / 5000

    x_matrix = x_over_z * z_matrix
    y_matrix = y_over_z * z_matrix
    return x_matrix, y_matrix, z_matrix


def get_normal(points):
    sum_points = [0, 0, 0]
    for point in points:
        sum_points += point
    c = sum_points / len(points)
    A = np.array(points) - c
    U, S, VT = np.linalg.svd(A)
    n = VT[2]
    return n[0], n[1], n[2], np.dot(n, c)


def main():
    color_matrix = cv2.imread('annot.png', cv2.IMREAD_COLOR)
    matrix_depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)

    rows, columns, channels = color_matrix.shape

    columns_indices = np.arange(0, columns)

    matrix_v = np.tile(columns_indices, (rows, 1))
    matrix_u = np.transpose(np.tile(np.arange(rows), (columns, 1)))

    x, y, z, = convert_from_plane_to_3d(matrix_u, matrix_v, matrix_depth, cx=319.50, cy=239.50, focal_x=481.20,
                                        focal_y=-480.00)  # getting xyz coordinates of each point

    matrix_xyz = np.dstack((x, y, z))
    answ_matrix = matrix_xyz.reshape(rows * columns, 3)  # now we have a list of points

    reshaped_color_matrix = color_matrix.reshape(-1, color_matrix.shape[2])  # reshape matrix in order to get unique colors
    colors_unique = np.unique(reshaped_color_matrix, axis=0)
    print(colors_unique)

    num_colors, channels = colors_unique.shape
    equations = np.zeros((num_colors, 4))

    pc = o3d.geometry.PointCloud()

    pc.points = o3d.utility.Vector3dVector(answ_matrix)
    pc.normals = o3d.utility.Vector3dVector(np.zeros_like(answ_matrix))

    num_of_points, xyz = reshaped_color_matrix.shape

    i = 0
    for color in colors_unique:
        if (color != [0, 0, 0]).all():  # for every unique color
            print('цвет')
            print(color)
            plane_points = []
            plane_points_indices = []
            k = 0
            for k in range(num_of_points):
                v = (reshaped_color_matrix[k] == color).all()   # finding points of specific color
                if v:
                    plane_points.append(answ_matrix[k])
                    plane_points_indices.append(k)
                k += 1
            plane_points_array = np.array(plane_points)
            equations[i] = np.array(get_normal(plane_points_array))  # getting equation out of plane

            normals = np.asarray(pc.normals)
            normals[np.asarray(plane_points_indices)] = 10 * equations[i][:-1]
            pc.normals = o3d.utility.Vector3dVector(normals)
            o3d.visualization.draw_geometries([pc],  point_show_normal=True)
            i += 1


if __name__ == '__main__':
    main()
