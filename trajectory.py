import sys
from typing import List

import numpy as np
import math


#from memory_profiler import profile
#from pypcd import pypcd
from pypcd import pypcd

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
    elif r00 > r11 and r00 > r22:
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


def image_processing(function, depth_annot, camera_intrinsics, map_indx_points, planes, func_indx, planes_matcher):
    points_of_images = []
    colors_of_images = []

    matrix_v = None
    i = 0
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

        if function != None:
            if func_indx == 1:
                function(points_of_image, colors_of_image, planes_matcher, map_indx_points)
            elif func_indx == 2:
                function(points_of_image, colors_of_image, planes_matcher, map_indx_points, planes)
            elif func_indx == 3:
                if i == 0:
                    indx, cur_planes = function(points_of_image, colors_of_image, map_indx_points, 0, None)
                    planes.append(cur_planes)
                else:
                    indx, cur_planes = function(points_of_image, colors_of_image, map_indx_points, indx, planes[i-1])
                    planes.append(cur_planes)
        i += 1

    return points_of_images, colors_of_images


def image_processing_office(function, colors_files, depths_files, camera_intrinsics,
                            map_indx_points,
                            planes,
                            planes_matcher,
                            first_or_sec):

        points_of_images = []
        colors_of_images = []

        for i, image in enumerate(colors_files):

            colors_of_image = cv2.imread(image, cv2.IMREAD_COLOR)
            colors_of_image = colors_of_image.reshape(-1, colors_of_image.shape[2])
            colors_of_images.append(colors_of_image)

            points_of_image = read_office.getting_points(i, depths_files, camera_intrinsics)
            points_of_images.append(points_of_image)
            print(image)

            if first_or_sec == 1:
                function(points_of_image, colors_of_image, planes_matcher, map_indx_points)
            elif first_or_sec == 2:
                function(points_of_image, colors_of_image, planes_matcher, map_indx_points, planes)

        return points_of_images, colors_of_images


def point_cloud_processing(data_directory, map_indx_points, planes, indx_to_color):
    folders = os.listdir(data_directory)

    points_of_images = []
    colors_of_images = []
    annot_of_images = []
    set_of_generated_colors = set()
    set_of_generated_colors.add("0#0#0")

    planes_global_indx = 0

    true_colors = []

    for j, folder in enumerate(folders):
        cur_path = os.path.join(data_directory, folder)
        npy, pcd = os.listdir(cur_path)

        print(npy)

        annot_of_image = np.load(os.path.join(cur_path, npy))
        path_to_pc = os.path.join(cur_path, pcd)
        pc = o3d.io.read_point_cloud(path_to_pc)
        pc_of_image = np.asarray(pc.points)/1000
        pc = None
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(pc_of_image)
        # o3d.visualization.draw_geometries([pc])


        colors_of_image = np.zeros(pc_of_image.shape)
        annot_unique = np.unique(annot_of_image, axis=0)

        # for indx in annot_unique:
        #     indices = np.where((annot_of_image == indx))
        #     cur_color = generate_color(set_of_generated_colors)
        #     if indx != 1:
        #         colors_of_image[indices[0]] = cur_color
        # colors_of_images.append(np.asarray(colors_of_image))



        # if function_indx == 1:
        #     building_maps_frontend(pc_of_image, annot_of_image, planes_matcher, map_indx_points)

        points_of_images.append(pc_of_image)
        annot_of_images.append(annot_of_image)
        colors_of_images.append(colors_of_image)

        if j == 0:
            planes_global_indx, cur_planes, cur_true_colors = planes_extraction_for_frontend(pc_of_image, annot_of_image,
                                                                             planes_global_indx, map_indx_points, None, indx_to_color, set_of_generated_colors)
            planes.append(cur_planes)
            true_colors.append(cur_true_colors)
            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(pc_of_image)
            # pc.colors = o3d.utility.Vector3dVector(cur_true_colors.astype(np.float64) / 255.0)
            # o3d.visualization.draw_geometries([pc])

        else:
            planes_global_indx, cur_planes, cur_true_colors = planes_extraction_for_frontend(pc_of_image, annot_of_image,
                                                                           planes_global_indx, map_indx_points, planes[j-1], indx_to_color, set_of_generated_colors)
            planes.append(cur_planes)
            true_colors.append(cur_true_colors)
            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(pc_of_image)
            # pc.colors = o3d.utility.Vector3dVector(cur_true_colors.astype(np.float64) / 255.0)
            # o3d.visualization.draw_geometries([pc])

        # o3d.visualization.draw_geometries([pc])
    return np.asarray(points_of_images), np.asarray(true_colors), np.asarray(annot_of_images), planes


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


def measure_error_orb3(first_node, first_gt_node, num_of_nodes, ds_filename_gt, filename_orb3, file_name_estimated,
                  file_name_gt):
    file_to_write_gt = open("measure_error_gt.txt", 'w')
    file_to_read_gt = open(ds_filename_gt)

    in_file = file_to_read_gt.readlines()  # mind the first timestamp

    for line in in_file[first_node: first_node + num_of_nodes]:
        file_to_write_gt.write(line + '\n')
    file_to_write_gt.close()

    file_to_write_estimated = open(file_name_estimated, 'w')
    file_to_read_estimated = open(filename_orb3)

    in_file = file_to_read_estimated.readlines()

    for line in in_file[first_node: first_node + num_of_nodes]:
        file_to_write_estimated.write(line + '\n')
    file_to_write_estimated.close()

    print("ate")
    evaluate_ate.main(file_name_gt, file_name_estimated)
    print("rpe")
    evaluate_rpe.main(file_name_gt, file_name_estimated)


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


#@profile
def visualisation(graph_estimated_state, num_of_nodes, matrices_of_points, colors):
    reflection = np.asarray(
        [[-1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]]
    )

    pc_answ = o3d.geometry.PointCloud()
    for i in range(num_of_nodes):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(matrices_of_points[len(matrices_of_points) - i - 1])
        pc.colors = o3d.utility.Vector3dVector(colors[len(matrices_of_points) - i - 1].astype(np.float64) / 255.0)
        pc = pc.transform(graph_estimated_state[-(i + 1)])
        matrices_of_points[len(matrices_of_points) - i - 1] = None
        pc = pc.voxel_down_sample(0.05) # free memory immediately as vector3dvector copies data (don't know how to prevent it)
        pc = pc.transform(reflection)
        pc_answ += pc

    pc_answ = pc_answ.voxel_down_sample(0.05)

    o3d.visualization.draw_geometries([pc_answ])

#@profile
def main(path_to_front):
    planes_matcher = PM
    map_indx_points = MAP_IP
    planes = []
    indx_to_color = {}

    directory = os.listdir(path_to_front)
    num_of_nodes = len(directory)

    matrices_of_points, colors, annot, planes = point_cloud_processing(path_to_front, map_indx_points, planes, indx_to_color)

    map_indx_points_sorted = sorted(map_indx_points, key=map_indx_points.get)
    max_indices = map_indx_points_sorted[-4:]

    cur_colors = np.zeros(colors.shape)
    planes_to_mrob = []
    cur_good_planes = []
    for i, planes_of_image in enumerate(planes[0:315]):
        for plane in planes_of_image:
            # cur_color = indx_to_color[plane.index]
            # indices = np.where((colors[i] == cur_color).all(axis=1))
            # cur_colors[i][indices[0]] = np.copy(colors[i][indices[0]])
            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(matrices_of_points[i])
            # pc.colors = o3d.utility.Vector3dVector(cur_colors[i].astype(np.float64) / 255.0)
            #o3d.visualization.draw_geometries([pc])
            if plane.index in max_indices:
                cur_good_planes.append(plane)
                cur_color = indx_to_color[plane.index]
                indices = np.where((colors[i] == cur_color).all(axis=1))
                cur_colors[i][indices[0]] = np.copy(colors[i][indices[0]])
        #if i >= 28:
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(matrices_of_points[i])
        # pc.colors = o3d.utility.Vector3dVector(cur_colors[i].astype(np.float64) / 255.0)
        # o3d.visualization.draw_geometries([pc])
        planes_to_mrob.append(cur_good_planes)
        cur_good_planes = []



    plane_index_to_real_index = {}

    graph, graph_trajectory = build_the_graph(num_of_nodes, planes_to_mrob, plane_index_to_real_index)
    graph_estimated_state = estimate_the_graph(graph, num_of_nodes, planes_to_mrob, plane_index_to_real_index, graph_trajectory)

    visualisation(graph_estimated_state, num_of_nodes, matrices_of_points, colors)


    #     for plane in planes_of_image:
    #         if plane.index not in max_indices:
    #             planes_of_image.remove(plane) # убираем все маленькие плоскости
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(mat)
    #     pc.colors = o3d.utility.Vector3dVector(cur_colors.astype(np.float64) / 255.0)




    # pahlava = os.listdir(path_color)
    # pahlava = sorted(pahlava, key=lambda x: int(x[:-4]))
    # #full_pahlava = list(map(lambda x: os.path.join(path_color, x), pahlava))
    #
    # num_of_nodes = len(pahlava)
    #
    # pahlava_orig = os.listdir(path_orig_color)
    # pahlava_orig = sorted(pahlava_orig, key=lambda x: int(x[:-4]))
    # #full_pahlava_orig = map(lambda x: os.path.join(path_orig_color, x), pahlava_orig)
    #
    # depth = os.listdir(path_depth)
    # depth = sorted(depth,  key=lambda x: int(x[:-4]))
    # #full_depth = list(map(lambda x: os.path.join(path_depth, x), depth))
    #
    # depth_orig = os.listdir(path_orig_depth)
    # depth_orig = sorted(depth_orig,  key=lambda x: int(x[:-4]))
    # #full_depth_orig = map(lambda x: os.path.join(path_orig_depth, x), pahlava_orig)
    #
    # depth_annot_orig = \
    #     [[os.path.join(path_orig_color, pahlava_orig[i]), os.path.join(path_orig_depth, depth_orig[i])] for i, _ in enumerate(depth_orig)]
    #
    # depth_annot = \
    #     [[os.path.join(path_color, pahlava[i]), os.path.join(path_depth, depth[i])] for i, _ in enumerate(depth)]

    #
    # planes = []
    # map_indx_points  = {}
    #
    # lambda_3 = lambda x, y, z, p, q: planes_extraction_for_frontend(x, y, z, p, q)
    # matrices_of_points, colors = image_processing(lambda_3, depth_annot, CAMERA_ICL, map_indx_points, planes, 3, planes_matcher=None) # в этой строчке получаем planes и карту (индекс: число точек)

    # planes_matcher = PM
    # map_indx_points = MAP_IP
    #
    # if len(planes_matcher) == 0:
    #     lambda_1 = lambda x, y, z, q: building_maps(x, y, z, q)
    #     image_processing_office(lambda_1, full_pahlava_orig, depths_files_orig, CAMERA_ICL, planes_matcher, map_indx_points, planes=None, first_or_sec=1)

    # map_indx_points_sorted = sorted(map_indx_points, key=map_indx_points.get)
    # max_indices = map_indx_points_sorted[-4:]

    # for planes_of_image in planes:
    #     for plane in planes_of_image:
    #         if plane.index not in max_indices:
    #             planes_of_image.remove(plane) # убираем все маленькие плоскости
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(mat)
    #     pc.colors = o3d.utility.Vector3dVector(cur_colors.astype(np.float64) / 255.0)
    #     o3d.visualization.draw_geometries([pc])
    #
    # lambda_2 = lambda x, y, z, q, p: equation_extraction(x, y, z, q, p)
    # matrices_of_points, colors = \
    #     image_processing_office(lambda_2, full_pahlava, depths_files, CAMERA_ICL, planes_matcher, max_indices, planes, 2)

    # plane_index_to_real_index = {}
    #
    # graph, graph_trajectory = build_the_graph(num_of_nodes, planes, plane_index_to_real_index)
    # graph_estimated_state = estimate_the_graph(graph, num_of_nodes, planes, plane_index_to_real_index, graph_trajectory)
    # num_of_nodes = 144
    #
    # measure_error_orb3(1365, 1, num_of_nodes, 'CameraTrajectory.txt',  'quaternion_gt_living_room.txt',
    #                    'measure_error_estimated.txt', 'measure_error_gt.txt')

    #visualisation(graph_estimated_state, num_of_nodes, matrices_of_points, colors)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build a trajectory')
    parser.add_argument('path_to_front', type=str, help='Directory where color images are stored')
    # parser.add_argument('path_orig_color', type=str, help='Directory where color images are stored')
    # #parser.add_argument('number_of_nodes', type=int, help='Number of nodes in the dataset')
    # parser.add_argument('path_depth', type=str, help='Directory, where all the files are stored')
    # parser.add_argument('path_orig_depth', type=str, help='Directory, where working files are stored')
    args = parser.parse_args()

    main(args.path_to_front)
