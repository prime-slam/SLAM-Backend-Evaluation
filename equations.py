import numpy as np
import math
import cv2
import open3d as o3d


def convert_from_plane_to_3d(u, v, depth, cx, cy, focal_x, focal_y):
    x_over_z = (cx - u) / focal_x
    y_over_z = (cy - v) / focal_y

    z_matrix = depth / 5000

    x_matrix = x_over_z * z_matrix
    y_matrix = y_over_z * z_matrix
    return x_matrix, y_matrix, z_matrix


def get_normal(points):
    c = np.mean(points, axis=0)
    A = np.array(points) - c
    eigvals, eigvects = np.linalg.eig(A.T@A)
    min_index = np.argmin(eigvals)
    n = eigvects[:, min_index]
    answ = np.asarray([n[0], n[1], n[2], -np.dot(n, c)])
    return answ


def main():
    color_matrix = cv2.imread('annot.png', cv2.IMREAD_COLOR)
    matrix_depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)

    rows, columns, _ = color_matrix.shape

    columns_indices = np.arange(columns)

    matrix_v = np.tile(columns_indices, (rows, 1))
    matrix_u = np.transpose(np.tile(np.arange(rows), (columns, 1)))

    x, y, z, = convert_from_plane_to_3d(
        matrix_u,
        matrix_v,
        matrix_depth,
        cx=319.50,
        cy=239.50,
        focal_x=481.20,
        focal_y=-480.00
    )  # getting xyz coordinates of each point

    matrix_xyz = np.dstack((x, y, z))
    matrix_of_points = matrix_xyz.reshape(-1, matrix_xyz.shape[2])  # now we have a list of points
    reshaped_color_matrix = color_matrix.reshape(-1, color_matrix.shape[2])  # reshape matrix in order to get unique colors

    colors_unique = np.unique(reshaped_color_matrix, axis=0)

    num_unique_colors, _ = colors_unique.shape
    equations = []

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(matrix_of_points)
    pc.normals = o3d.utility.Vector3dVector(np.zeros_like(matrix_of_points))

    num_of_points, _ = reshaped_color_matrix.shape

    unique_colors_without_black = filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)

    for i, color in enumerate(unique_colors_without_black):
        indices = np.where((reshaped_color_matrix == color).all(axis=1))
        plane_points = matrix_of_points[indices[0]]

        plane_points_array = np.array(plane_points)
        equations.append(get_normal(plane_points_array))  # getting equation out of plane

        normals = np.asarray(pc.normals)
        normals[np.asarray(indices)] =  10 * equations[i][:-1]
        pc.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([pc],  point_show_normal=True)

if __name__ == '__main__':
    main()