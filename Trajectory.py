from typing import List

import numpy as np
import math
from config import CAMERA_ICL
import cv2
from plane_extraction import *
import open3d as o3d
import os
import argparse
from mrob.mrob import FGraph, geometry, registration, LM


def image_processing(path_color, path_depth, camera_intrinsics):
    matrix_color = cv2.imread(path_color, cv2.IMREAD_COLOR)
    matrix_depth = cv2.imread(path_depth, cv2.IMREAD_ANYDEPTH)

    rows, columns, _ = matrix_color.shape
    columns_indices = np.arange(columns)

    matrix_v = np.tile(columns_indices, (rows, 1))
    matrix_u = np.transpose(np.tile(np.arange(rows), (columns, 1)))

    matrix_xyz = convert_from_plane_to_3d(
        matrix_u,
        matrix_v,
        matrix_depth,
        camera_intrinsics
    )  # getting xyz coordinates of each point

    matrix_of_points = matrix_xyz.reshape(-1, matrix_xyz.shape[2])  # now we have a list of points
    reshaped_color_matrix = matrix_color.reshape(-1, matrix_color.shape[2])
    return matrix_of_points, reshaped_color_matrix


def read_gt_poses(path, dataset_nodes):
    in_file = open(path).read().splitlines()  # getting gt data

    all_lines = filter(lambda line: (line != ''), in_file)

    array_with_lines = np.loadtxt(all_lines)
    gt_matrices = np.split(array_with_lines, dataset_nodes, axis=0)
    return gt_matrices


def measure_error(num_of_nodes, graph_estimated_state, dataset_nodes):

    gt_matrices = read_gt_poses('gt.txt', dataset_nodes)
    last_gt_matrix = gt_matrices[num_of_nodes - 1]  # measuring error only by the last matrix
    gt_matrix_01 = gt_matrices[0]

    matrix_gt_the_last = np.append(last_gt_matrix, [[0, 0, 0, 1]], axis=0)
    matrix_gt_first = np.append(gt_matrix_01, [[0, 0, 0, 1]], axis=0)

    the_last_estimated_node = geometry.SE3(graph_estimated_state[-1])

    absolute_to_the_last_gt = np.linalg.inv(matrix_gt_first) @ matrix_gt_the_last

    matrix_gt_se3 = geometry.SE3(absolute_to_the_last_gt)

    error_rotation = the_last_estimated_node.distance_rotation(matrix_gt_se3)
    error_translation = the_last_estimated_node.distance_trans(matrix_gt_se3)

    return error_rotation, error_translation


def estimate_the_graph(graph, num_of_nodes, planes, plane_index_to_real_index):

    w_z = np.identity(4)    # weight matrix

    graph_trajectory = []

    for i in range(num_of_nodes):
        next_node = graph.add_node_pose_3d(geometry.SE3())
        graph_trajectory.append(next_node)
    graph.add_factor_1pose_3d(geometry.SE3(), graph_trajectory[0], 1e6 * np.identity(6))

    for i in range(num_of_nodes):
        real_indx = len(plane_index_to_real_index)
        for plane in planes[i]:
            if plane.index not in plane_index_to_real_index:
                plane_index_to_real_index[plane.index] = real_indx
                real_indx += 1

    for n in range(num_of_nodes):
        for plane in planes[n]:
            cur_indx = plane_index_to_real_index[plane.index]
            graph.add_factor_1pose_1plane_4d(plane.equation, graph_trajectory[n], cur_indx, w_z)

    graph.solve(LM)
    graph_estimated_state = graph.get_estimated_state()

    return graph_estimated_state


def build_the_graph(num_of_nodes, planes: List[List[Plane]]):
    graph = FGraph()
    set_indx = set()

    for i in range(num_of_nodes):
        for plane in planes[i]:
            if plane.index not in set_indx:
                graph.add_node_plane_4d(
                    np.array([1, 0, 0, 0]))
                set_indx.add(plane.index)

    return graph


def visualisation(graph_estimaed_state, num_of_nodes, matrices_of_points, colors):

    point_clouds = []

    for i in range(num_of_nodes):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(matrices_of_points[i])
        pc.colors = o3d.utility.Vector3dVector(colors[i].astype(np.float64) / 255.0)
        point_clouds.append(pc)

    reflection = np.asarray(
        [[-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]]
    )

    pc_answ = o3d.geometry.PointCloud()
    map_cloud_matrix = {point_clouds: graph_estimaed_state for point_clouds, graph_estimaed_state in zip(reversed(point_clouds), reversed(graph_estimaed_state))}

    for pc in point_clouds:
        pc_answ += pc.transform(map_cloud_matrix[pc]).transform(reflection)
    pc_answ = pc_answ.voxel_down_sample(0.01)

    o3d.visualization.draw_geometries([pc_answ])


def main(path_depth, path_color, dataset_nodes):

    depths = sorted(os.listdir(path_depth), key=lambda x: int(x[:-4]))
    pahlava = sorted(os.listdir(path_color), key=lambda x: int(x[:-4]))

    num_of_nodes = len(depths)

    depth_annot =\
        [[os.path.join(path_color, pahlava[i]), os.path.join(path_depth, depths[i])] for i, _ in enumerate(depths)]

    planes = []
    matrices_of_points = []
    colors = []

    map_indx_points = {}
    planes_matcher = {}

    for image, depth in depth_annot:
        points_of_image, colors_of_image = image_processing(image, depth, CAMERA_ICL)
        building_maps(points_of_image, colors_of_image, planes_matcher, map_indx_points)

    max_points_indx = {}

    for key, value in map_indx_points.items():
        max_points_indx[value] = key

    list_of_max_points = list(map_indx_points.values())
    ten_max_points = sorted(list_of_max_points)[-10:]

    ten_max_indices = []

    for points in ten_max_points:
        ten_max_indices.append(max_points_indx[points])

    for image, depth in depth_annot:
        points_of_image, colors_of_image = image_processing(image, depth, CAMERA_ICL)
        planes_of_image = equation_extraction(points_of_image, colors_of_image, planes_matcher, ten_max_indices)

        matrices_of_points.append(points_of_image)
        colors.append(colors_of_image)
        planes.append(planes_of_image)

    plane_index_to_real_index = {}

    graph = build_the_graph(num_of_nodes, planes)
    graph_estimated_state = estimate_the_graph(graph, num_of_nodes, planes, plane_index_to_real_index)

    error_rotation, error_translation = measure_error(num_of_nodes, graph_estimated_state, dataset_nodes)
    print(error_rotation, error_translation)

    visualisation(graph_estimated_state, num_of_nodes, matrices_of_points, colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a trajectory')
    parser.add_argument('path_depth', type=str, help='Directory, where depth images are stored')
    parser.add_argument('path_color', type=str, help='Directory where color images are stored')
    parser.add_argument('number_of_nodes', type=int, help='Number of nodes in the dataset')

    args = parser.parse_args()

    main(args.path_depth, args.path_color, args.number_of_nodes)
    
