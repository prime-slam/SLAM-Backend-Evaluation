import numpy as np
import math
import cv2
import plane_extraction
import open3d as o3d
import os
from mrob.mrob import FGraph, geometry, registration, LM


def image_processing(path_color, path_depth):
    color_matrix = cv2.imread(path_color, cv2.IMREAD_COLOR)
    matrix_depth = cv2.imread(path_depth, cv2.IMREAD_ANYDEPTH)

    rows, columns, _ = color_matrix.shape

    columns_indices = np.arange(columns)

    matrix_v = np.tile(columns_indices, (rows, 1))
    matrix_u = np.transpose(np.tile(np.arange(rows), (columns, 1)))

    camera_intrinsics = plane_extraction.Camera(cx=319.50, cy=239.50, focal_x=481.20, focal_y=-480.00)

    matrix_xyz = plane_extraction.convert_from_plane_to_3d(
        matrix_u,
        matrix_v,
        matrix_depth,
        camera_intrinsics
    )  # getting xyz coordinates of each point

    matrix_of_points = matrix_xyz.reshape(-1, matrix_xyz.shape[2])  # now we have a list of points
    reshaped_color_matrix = color_matrix.reshape(-1,
                                                 color_matrix.shape[2])  # reshape matrix in order to get unique colors
    return matrix_of_points, reshaped_color_matrix


def main():

    planes_matcher = {}  # map (color: index)
    path_depth = "depth/200/"
    path_color = "markup/200/"

    depths = sorted(os.listdir(path_depth), key=lambda x: int(x.replace(".png", "")))
    pahlava = sorted(os.listdir(path_color),   key=lambda x: int(x.replace(".png", "")))

    num_of_nodes = len(depths)

    depth_annot = []
    for i in range(len(depths)):
        depth_annot.append(["markup/200/" + pahlava[i], "depth/200/" + depths[i]])

    equations = []
    matrices_of_points = []
    colors = []

    for image, depth in depth_annot:
        points_of_image, colors_of_image = image_processing(image, depth)
        equations_of_image = plane_extraction.equation_extraction(points_of_image, colors_of_image, planes_matcher)

        matrices_of_points.append(points_of_image)
        colors.append(colors_of_image)
        equations.append(equations_of_image)

    point_clouds = []

    for i in range(num_of_nodes):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(matrices_of_points[i])
        pc.colors = o3d.utility.Vector3dVector(colors[i].astype(np.float64) / 255.0)
        point_clouds.append(pc)
        # pc_2.normals = o3d.utility.Vector3dVector(np.zeros_like(matrix_of_points_2))

    graph = FGraph()
    set_indx = set()

    for i in range(num_of_nodes):
        for _, index in equations[i]:
            if index not in set_indx:
                graph.add_node_plane_4d(
                    np.array([1, 0, 0, 0]))
                set_indx.add(index)
    W_z = np.identity(4)    # weight matrix

    graph_trajectory = []

    for i in range(num_of_nodes):
        next_node = graph.add_node_pose_3d(geometry.SE3())
        graph_trajectory.append(next_node)
    graph.add_factor_1pose_3d(geometry.SE3(), graph_trajectory[0], 1e6 * np.identity(6))

    for n in range(num_of_nodes):
        for equation in equations[n]:
            graph.add_factor_1pose_1plane_4d(equation[0], graph_trajectory[n], equation[1], W_z)

    graph.solve(LM)
    x = graph.get_estimated_state()

    all_lines = []

    in_file = open('gt.txt').read().splitlines()
    for line in in_file:
        if line != '':
            all_lines.append(line)

    array_with_lines = np.loadtxt(all_lines)
    gt_matrices = np.split(array_with_lines, 1508, axis=0)

    gt_translation_0 = [[0, 0, -2.25]]
    q_0 = [0, 0, 0, 1]

    gt_matrix_200 = gt_matrices[199]
    gt_matrix_01 = gt_matrices[0]

    reflection = np.asarray(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    T1 = geometry.SE3(x[-1])
    print("T1")
    print(T1.T())
    print()

    T_gt_200 = np.append(gt_matrix_200, [[0, 0, 0, 1]], axis=0)
    T_gt_01 = np.append(gt_matrix_01, [[0, 0, 0, 1]], axis=0)

    transation_gt = (np.linalg.inv(T_gt_01) @ T_gt_200 - np.linalg.inv(T_gt_01) @ T_gt_200)[:3, 3]
    print(transation_gt)

    pc_answ = o3d.geometry.PointCloud()

    for i, pc in enumerate(reversed(point_clouds)):
        pc_answ += pc.transform(x[-(i + 1)]).transform(reflection)

    o3d.visualization.draw_geometries([pc_answ])

    absolute_gt = np.linalg.inv(T_gt_01) @ T_gt_200
    print('T_gt * T_gt_0^(-1))')
    print(absolute_gt)
    print()

    T_gt_SE3 = geometry.SE3(absolute_gt)

    print(T1.distance_rotation(T_gt_SE3))
    print('############')
    print(T1.distance_trans(T_gt_SE3))


if __name__ == '__main__':
    main()
