import sys
from typing import List

import numpy as np
import math
from config import CAMERA_ICL, PM, MAP_IP
import cv2
import read_office
import evaluate_ate
import evaluate_rpe
from plane_extraction import *
import open3d as o3d
import os
import subprocess as sp
import argparse
from mrob.mrob import FGraph, geometry, registration, LM


def rotation_matrix_to_quaternion(r_matrix):
    # First row of the rotation matrix
    r00 = r_matrix[0, 0]
    r01 = r_matrix[0, 1]
    r02 = r_matrix[0, 2]

    # Second row of the rotation matrix
    r10 = r_matrix[1, 0]
    r11 = r_matrix[1, 1]
    r12 = r_matrix[1, 2]

    # Third row of the rotation matrix
    r20 = r_matrix[2, 0]
    r21 = r_matrix[2, 1]
    r22 = r_matrix[2, 2]

    tr = r00 + r11 + r22

    if tr > 0:
        s = math.sqrt(tr+1.0) * 2
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif r00 > r11 & r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s

    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    q = [qx, qy, qz, qw]

    return q


def image_processing(function, depth_annot, camera_intrinsics, planes_matcher, map_indx_points, planes, first_or_sec):
    points_of_images = []
    colors_of_images = []

    matrix_v = None

    for image, depth in depth_annot:
        matrix_color = cv2.imread(image, cv2.IMREAD_COLOR)
        matrix_depth = cv2.imread(depth, cv2.IMREAD_ANYDEPTH)
        print(image)
        if matrix_v is None:
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

        points_of_image = matrix_xyz.reshape(-1, matrix_xyz.shape[2])  # now we have a list of points
        colors_of_image = matrix_color.reshape(-1, matrix_color.shape[2])

        points_of_images.append(points_of_image)
        colors_of_images.append(colors_of_image)

        if first_or_sec == 1:
            function(points_of_image, colors_of_image, planes_matcher, map_indx_points)
        elif first_or_sec == 2:
            function(points_of_image, colors_of_image, planes_matcher, map_indx_points, planes)

    return points_of_images, colors_of_images


def image_processing_office(function, colors_files, depths_files, camera_intrinsics, planes_matcher,
                            map_indx_points,
                            planes,
                            first_or_sec):

        points_of_images = []
        colors_of_images = []

        for i, image in enumerate(colors_files):

            colors_of_image = cv2.imread(image, cv2.IMREAD_COLOR)
            colors_of_image = colors_of_image.reshape(-1, colors_of_image.shape[2])
            colors_of_images.append(colors_of_image)

            points_of_image = read_office.getting_points(i , depths_files, camera_intrinsics)
            points_of_images.append(points_of_image)
            print(image)

            if first_or_sec == 1:
                function(points_of_image, colors_of_image, planes_matcher, map_indx_points)
            elif first_or_sec == 2:
                function(points_of_image, colors_of_image, planes_matcher, map_indx_points, planes)

        return points_of_images, colors_of_images


def read_gt_poses(path, dataset_nodes):
    in_file = open(path).readlines()  # getting gt data

    all_lines = filter(lambda line: (line != ''), in_file)

    array_with_lines = np.loadtxt(all_lines)
    gt_matrices = np.split(array_with_lines, dataset_nodes, axis=0)
    return gt_matrices


def make_a_string(timestamp, translation, rotation):
    translation_str = ' '.join(str(e) for e in translation)
    rotation_str = ' '.join(str(e) for e in rotation)
    return str(timestamp) + ' ' + translation_str + ' ' + rotation_str


def measure_error(first_node, first_gt_node, num_of_nodes, graph_estimated_state, ds_filename_gt, file_name_estimated,
                  file_name_gt):

    file_to_write_gt = open("measure_error_gt.txt", 'w')
    file_to_read_gt = open(ds_filename_gt)

    bios = 0
    bios_gt = 0

    in_file = file_to_read_gt.readlines()  # mind the first timestamp

    if first_node == 0 and first_gt_node > 0:
        bios = first_gt_node
    elif first_gt_node > 0:
        bios_gt = first_gt_node

    for line in in_file[first_node - bios_gt:first_node + num_of_nodes - bios_gt]:
        file_to_write_gt.write(line + '\n')
    file_to_write_gt.close()

    file_to_write_estimated = open(file_name_estimated, 'w')

    estimated_matrices = graph_estimated_state[-num_of_nodes+bios:]
    for i, matrix in enumerate(estimated_matrices):
        data = make_a_string(first_node + i + bios, matrix[:3, 3], rotation_matrix_to_quaternion(matrix[:3, :3]))
        file_to_write_estimated.write(data + '\n')
    file_to_write_estimated.close()

    print("ate")
    evaluate_ate.main(file_name_gt, file_name_estimated)
    print("rpe")
    evaluate_rpe.main(file_name_gt, file_name_estimated)


def estimate_the_graph(graph, num_of_nodes, planes, plane_index_to_real_index, graph_trajectory):

    w_z = np.identity(4)    # weight matrix

    for n in range(num_of_nodes):
        for plane in planes[n]:
            cur_indx = plane_index_to_real_index[plane.index]
            graph.add_factor_1pose_1plane_4d(plane.equation, graph_trajectory[n], cur_indx, w_z)

    graph.solve(LM)
    graph_estimated_state = graph.get_estimated_state()

    return graph_estimated_state


def build_the_graph(num_of_nodes, planes: List[List[Plane]], plane_index_to_real_index):
    graph = FGraph()

    for i in range(num_of_nodes):
        real_indx = len(plane_index_to_real_index)
        for plane in planes[i]:
            if plane.index not in plane_index_to_real_index:
                plane_index_to_real_index[plane.index] = real_indx
                real_indx += 1

    for i, _ in enumerate(plane_index_to_real_index):
        graph.add_node_plane_4d(
            np.array([1, 0, 0, 0]))

    graph_trajectory = []

    for i in range(num_of_nodes):
        next_node = graph.add_node_pose_3d(geometry.SE3())
        graph_trajectory.append(next_node)
    graph.add_factor_1pose_3d(geometry.SE3(), graph_trajectory[0], 1e6 * np.identity(6))

    return graph, graph_trajectory


def visualisation(graph_estimated_state, num_of_nodes, matrices_of_points, colors):

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
    map_cloud_matrix = {point_clouds: graph_estimated_state for point_clouds, graph_estimated_state in zip(reversed(point_clouds), reversed(graph_estimated_state))}

    for pc in point_clouds:
        pc_answ += pc.transform(map_cloud_matrix[pc]).transform(reflection)
    pc_answ = pc_answ.voxel_down_sample(0.01)

    o3d.visualization.draw_geometries([pc_answ])


def main(path_color, path_orig_color, num_of_nodes, general_path_orig, general_path):

    pahlava = os.listdir(path_color)
    pahlava = sorted(pahlava, key=read_office.__filenames_sorted_mapper)
    full_pahlava = list(map(lambda x: os.path.join(path_color, x), pahlava))

    pahlava_orig = os.listdir(path_orig_color)
    pahlava_orig = sorted(pahlava_orig, key=read_office.__filenames_sorted_mapper)
    full_pahlava_orig = map(lambda x: os.path.join(path_orig_color, x), pahlava_orig)

    png_files, depths_files = read_office.provide_filenames(general_path)
    png_files_orig, depths_files_orig = read_office.provide_filenames(general_path_orig)

    planes = []

    planes_matcher = PM
    map_indx_points = MAP_IP

    if len(planes_matcher) == 0:
        lambda_1 = lambda x, y, z, q: building_maps(x, y, z, q)
        image_processing_office(lambda_1, full_pahlava_orig, depths_files_orig, CAMERA_ICL, planes_matcher, map_indx_points, planes=None, first_or_sec=1)

    map_indx_points_sorted = sorted(map_indx_points, key=map_indx_points.get)
    max_indices = map_indx_points_sorted[-4:]

    lambda_2 = lambda x, y, z, q, p: equation_extraction(x, y, z, q, p)
    matrices_of_points, colors = \
        image_processing_office(lambda_2, full_pahlava, depths_files, CAMERA_ICL, planes_matcher, max_indices, planes, 2)

    plane_index_to_real_index = {}

    graph, graph_trajectory = build_the_graph(num_of_nodes, planes, plane_index_to_real_index)
    graph_estimated_state = estimate_the_graph(graph, num_of_nodes, planes, plane_index_to_real_index, graph_trajectory)

    measure_error(204, 2, num_of_nodes, graph_estimated_state, 'quaternion_gt_office.txt', 'measure_error_estimated.txt',
                  'measure_error_gt.txt')

    visualisation(graph_estimated_state, num_of_nodes, matrices_of_points, colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a trajectory')
    parser.add_argument('path_color', type=str, help='Directory where color images are stored')
    parser.add_argument('path_orig_color', type=str, help='Directory where color images are stored')
    parser.add_argument('number_of_nodes', type=int, help='Number of nodes in the dataset')
    parser.add_argument('general_path_orig', type=str, help='Directory, where all the files are stored')
    parser.add_argument('general_path', type=str, help='Directory, where working files are stored')

    args = parser.parse_args()

    main(args.path_color, args.path_orig_color, args.number_of_nodes, args.general_path_orig, args.general_path)
    