import numpy as np
import math
import cv2
import open3d as o3d
from mrob.mrob import FGraph, geometry, registration, LM


def color_to_string(array):
    string = ""
    for i in range(len(array)):
        string += str(array[i])
        string += "#"
    return string


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


def equation_extraction(points, point_colors, array_of_colors):
    colors_unique = np.unique(point_colors, axis=0)

    num_unique_colors, _ = colors_unique.shape
    equations = []

    num_of_points, _ = points.shape

    unique_colors_without_black = filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)

    if len(array_of_colors) == 0:   # if the map is empty, append all the colors from a picture
        indx = 0
        for color in unique_colors_without_black:
            color_string = color_to_string(color)
            array_of_colors.update({color_string: indx})
            indx += 1
    else:
        indx = len(array_of_colors)
        for color in unique_colors_without_black:
            color_string = color_to_string(color)
            if array_of_colors.get(color_string, 'none') == 'none':  # if the plane is new
                array_of_colors.update({color_string: indx})    # append (color:index) to map with the next index
                indx += 1

    for i, color in enumerate(unique_colors_without_black):
        color_string = color_to_string(color)
        indices = np.where((point_colors == color).all(axis=1))
        plane_points = points[indices[0]]

        plane_points_array = np.array(plane_points)
        equations.append((get_normal(plane_points_array), array_of_colors[color_string]))
    return equations


def image_processing(path_color, path_depth):
    color_matrix = cv2.imread(path_color, cv2.IMREAD_COLOR)
    matrix_depth = cv2.imread(path_depth, cv2.IMREAD_ANYDEPTH)

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
    reshaped_color_matrix = color_matrix.reshape(-1,
                                                 color_matrix.shape[2])  # reshape matrix in order to get unique colors
    return matrix_of_points, reshaped_color_matrix


def main():

    planes_matcher = {}  # map (color: index)

    num_of_nodes = 2

    equations = []

    matrix_of_points_1, colors_1 = image_processing('0000.png', 'depth_0000.png')
    equations_1 = equation_extraction(matrix_of_points_1, colors_1, planes_matcher)

    matrix_of_points_2, colors_2 = image_processing('0050.png', 'depth_0050.png')
    equations_2 = equation_extraction(matrix_of_points_2, colors_2, planes_matcher)

    pc_1 = o3d.geometry.PointCloud()
    pc_1.points = o3d.utility.Vector3dVector(matrix_of_points_1)
    pc_1.colors = o3d.utility.Vector3dVector(colors_1.astype(np.float64) / 255.0)
    # pc_1.normals = o3d.utility.Vector3dVector(np.zeros_like(matrix_of_points_1))

    pc_2 = o3d.geometry.PointCloud()
    pc_2.points = o3d.utility.Vector3dVector(matrix_of_points_2)
    pc_2.colors = o3d.utility.Vector3dVector(colors_2.astype(np.float64) / 255.0)
    # pc_2.normals = o3d.utility.Vector3dVector(np.zeros_like(matrix_of_points_2))

    equations.append(equations_1)
    equations.append(equations_2)

    graph = FGraph()
    set_of_indices = set()

    for i in range(num_of_nodes):
        for _, index in equations[i]:
            if index not in set_of_indices:     # add all the landmarks to the graph
                graph.add_node_plane_4d(
                    np.array([1, 0, 0, 0]))  # nodes are numbered from 0. No other node should be added BEFORE
                set_of_indices.add(index)
    W_z = np.identity(4)    # weight matrix

    graph_trajectory = []

    for i in range(num_of_nodes):
        n1 = graph.add_node_pose_3d(geometry.SE3())
        graph_trajectory.append(n1)
    graph.add_factor_1pose_3d(geometry.SE3(), graph_trajectory[0], 1e6 * np.identity(6))

    for n in range(num_of_nodes):
        equations[n] = list(map(lambda x: (x[0] * np.sign(x[0][-1]), x[1]), equations[n]))

    for n in range(num_of_nodes):
        for equation in equations[n]:
            graph.add_factor_1pose_1plane_4d(equation[0], graph_trajectory[n], equation[1], W_z)

    graph.solve(LM)
    x = graph.get_estimated_state()
    T1 = geometry.SE3(x[-1])

    pc_answ = pc_2.transform(x[-1])
    pc_1 = pc_1.transform(x[-2])
    pc_answ += pc_1

    gt_translation = [[0.129723, 0.00959134, - 2.25525]]
    q = [- 0.0051396, 0.0821083, 0.0461804, 0.995539]

    gt_rotation = np.asarray([[2 * (q[0] * q[0] + q[1] * q[1]) - 1, 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
                [2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[0] * q[0] + q[2] * q[2]) - 1,  2 * (q[2] * q[3] + q[0] * q[1])],
                [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 2 * (q[0] * q[0] + q[3] * q[3]) - 1]])  # ground truth matrix of rotation

    rotation_translation = np.append(gt_rotation, gt_translation, axis=0)
    T_gt = np.append(rotation_translation.T, [[0, 0, 0, 1]], axis=0)

    T_gt_SE3 = geometry.SE3(T_gt)

    print(T1.distance_rotation(T_gt_SE3))
    print('############')
    print(T1.distance_trans(T_gt_SE3))


if __name__ == '__main__':
    main()