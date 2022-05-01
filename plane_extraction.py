import numpy as np
import math
import cv2
import open3d as o3d


class Camera(object):
    def __init__(self, cx, cy, focal_x, focal_y, scale):
        self.cx = cx
        self.cy = cy
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.scale = scale


class Plane(object):
    def __init__(self, equation, indx):
        self.equation = equation
        self.index = indx


def array_to_string(array: np.array) -> str:

    channels = [str(num) for num in array]
    string = "#".join(channels)

    return string


def convert_from_plane_to_3d(u, v, depth, cam: Camera):
    x_over_z = (v - cam.cx) / cam.focal_x
    y_over_z = (u - cam.cy) / cam.focal_y

    z_matrix = depth / cam.scale

    x_matrix = x_over_z * z_matrix
    y_matrix = y_over_z * z_matrix

    return np.dstack((x_matrix, y_matrix, z_matrix))


def get_normal(points):
    c = np.mean(points, axis=0)
    A = np.array(points) - c
    eigvals, eigvects = np.linalg.eig(A.T@A)
    min_index = np.argmin(eigvals)
    n = eigvects[:, min_index]

    d = -np.dot(n, c)
    normal = int(np.sign(d)) * n
    d *= np.sign(d)
    return np.asarray([normal[0], normal[1], normal[2], d])


def building_maps(points, point_colors, color_to_index, indx_to_max_num_points):

    colors_unique = np.unique(point_colors, axis=0)

    num_unique_colors, _ = colors_unique.shape
    num_of_points, _ = points.shape

    unique_colors_without_black = filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)

    indx = len(color_to_index)
    for color in unique_colors_without_black:
        color_string = array_to_string(color)
        if color_string not in color_to_index:  # if the plane is new
            color_to_index[color_string] = indx  # append (color:index) to map with the next index
            indx += 1
        cur_indx = color_to_index[color_string]
        indices = np.where((point_colors == color).all(axis=1))
        if (cur_indx in indx_to_max_num_points and len(indices[0]) > indx_to_max_num_points[cur_indx]) \
                or cur_indx not in indx_to_max_num_points:
            indx_to_max_num_points[cur_indx] = len(indices[0])

    return color_to_index, indx_to_max_num_points


def equation_extraction(points, point_colors, color_to_index, max_planes):
    colors_unique = np.unique(point_colors, axis=0)

    planes = []
    unique_colors_without_black = list(filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique))

    for color in unique_colors_without_black:
        color_string = array_to_string(color)
        cur_indx = color_to_index[color_string]
        indices = np.where((point_colors == color).all(axis=1))
        if cur_indx in max_planes:
            plane_points = points[indices[0]]
            plane = Plane(get_normal(plane_points), color_to_index[color_string])
            planes.append(plane)

    return planes
