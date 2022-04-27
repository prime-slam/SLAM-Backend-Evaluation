import numpy as np
import math
import cv2
import plane_extraction
import open3d as o3d
import os
from mrob.mrob import FGraph, geometry, registration, LM


def image_processing(path_color, path_depth):
    matrix_color = cv2.imread(path_color, cv2.IMREAD_COLOR)
    matrix_depth = cv2.imread(path_depth, cv2.IMREAD_ANYDEPTH)

    rows, columns, _ = matrix_color.shape
    columns_indices = np.arange(columns)

    matrix_v = np.tile(columns_indices, (rows, 1))
    matrix_u = np.transpose(np.tile(np.arange(rows), (columns, 1)))

    camera_intrinsics = plane_extraction.Camera(cx=319.50, cy=239.50, focal_x=481.20, focal_y=-480.00, scale=5000)

    matrix_xyz = plane_extraction.convert_from_plane_to_3d(
        matrix_u,
        matrix_v,
        matrix_depth,
        camera_intrinsics
    )  # getting xyz coordinates of each point

    matrix_of_points = matrix_xyz.reshape(-1, matrix_xyz.shape[2])  # now we have a list of points
    reshaped_color_matrix = matrix_color.reshape(-1, matrix_color.shape[2])  # reshape matrix in order to get unique colors
    return matrix_of_points, reshaped_color_matrix


def read_gt(path):
    all_lines = []

    in_file = open(path).read().splitlines()  # getting gt data
    for line in in_file:
        if line != '':
            all_lines.append(line)

    array_with_lines = np.loadtxt(all_lines)
    gt_matrices = np.split(array_with_lines, len(os.listdir()), axis=0)
    return gt_matrices


def measuring_error(num_of_nodes, graph_estimated_state):

    gt_matrices = read_gt('gt.txt')
    gt_matrix_the_last = gt_matrices[num_of_nodes-1]  # measuring error only by the last matrix
    gt_matrix_01 = gt_matrices[0]

    T_gt_the_last = np.append(gt_matrix_the_last, [[0, 0, 0, 1]], axis=0)
    T_gt_01 = np.append(gt_matrix_01, [[0, 0, 0, 1]], axis=0)

    T1 = geometry.SE3(graph_estimated_state[-1])

    absolute_gt = np.linalg.inv(T_gt_01) @ T_gt_the_last

    T_gt_SE3 = geometry.SE3(absolute_gt)

    error_rotation = T1.distance_rotation(T_gt_SE3)
    error_translation = T1.distance_trans(T_gt_SE3)

    return error_rotation, error_translation


def work_with_graph(num_of_nodes, equations):
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

    map_index_of_plane_real_index = {}
    for i in range(num_of_nodes):
        real_indx = len(map_index_of_plane_real_index)
        for _, index in equations[i]:
            if index not in map_index_of_plane_real_index:
                map_index_of_plane_real_index[index] = real_indx
                real_indx += 1

    for n in range(num_of_nodes):
        for equation in equations[n]:
            cur_indx = map_index_of_plane_real_index[equation[1]]
            graph.add_factor_1pose_1plane_4d(equation[0], graph_trajectory[n], cur_indx, W_z)

    graph.solve(LM)
    graph_estimated_state = graph.get_estimated_state()

    error_rotation, error_translation = measuring_error(num_of_nodes, graph_estimated_state)

    return graph_estimated_state

def main():

    path_depth = "depth/50-100/"
    path_color = "markup/50-100/"

    depths = sorted(os.listdir(path_depth), key=lambda x: int(x[:-4]))
    pahlava = sorted(os.listdir(path_color), key=lambda x: int(x[:-4]))

    num_of_nodes = len(depths)

    depth_annot = []
    for i, _ in enumerate(depths):
        depth_annot.append((["markup/50-100/" + pahlava[i], "depth/50-100/" + depths[i]]))

    equations = []
    matrices_of_points = []
    colors = []
    map_indx_points = {}

    for image, depth in depth_annot:
        points_of_image, colors_of_image = image_processing(image, depth)
        planes_matcher, map_indx_points = plane_extraction.building_maps(points_of_image, colors_of_image)

    viceversa = {} # and one more map (max_num_points: indx)

    for key, value in map_indx_points.items():
        viceversa[value] = key

    list_of_max_points = list(map_indx_points.values())
    ten_max_points = sorted(list_of_max_points)[-10:]

    for points in ten_max_points:
       ten_max_indices = viceversa[points]

    for image, depth in depth_annot:
        points_of_image, colors_of_image = image_processing(image, depth)
        equations_of_image = plane_extraction.equation_extraction(points_of_image, colors_of_image, planes_matcher, ten_max_indices)
        # we take equations only of 10 maximum planes

        matrices_of_points.append(points_of_image)
        colors.append(colors_of_image)
        equations.append(equations_of_image)

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

    graph_estimaed_state = work_with_graph(num_of_nodes, equations)

    pc_answ = o3d.geometry.PointCloud()
    for i, pc in enumerate(reversed(point_clouds)):
        pc_answ += pc.transform(graph_estimaed_state[-(i + 1)]).transform(reflection)
    pc_answ = pc_answ.voxel_down_sample(0.01)

    o3d.visualization.draw_geometries([pc_answ])


if __name__ == '__main__':
    main()
    
