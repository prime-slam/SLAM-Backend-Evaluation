import numpy as np
import math
import cv2

class Camera(object):
    def __init__(self, cx, cy, focal_x, focal_y):
        """Constructor"""
        self.cx = cx
        self.cy = cy
        self.focal_x = focal_x
        self.focal_y = focal_y


def array_to_string(array: np.array)->str:
    channels = []
    for num in array:
        channels.append(str(num))
    string = "#".join(channels)
    return string


def convert_from_plane_to_3d(u, v, depth, cam: Camera):
    x_over_z = (v - cam.cx) / cam.focal_x
    y_over_z = (u - cam.cy) / cam.focal_y

    z_matrix = depth / 5000

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
    normal = (int)(np.sign(d)) * n
    d *= np.sign(d)
    return np.asarray([normal[0], normal[1], normal[2], d])


def equation_extraction(points, point_colors, map_color_index):
    colors_unique = np.unique(point_colors, axis=0)

    num_unique_colors, _ = colors_unique.shape
    equations = []

    num_of_points, _ = points.shape

    unique_colors_without_black = filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)

    indx = len(map_color_index)
    for color in unique_colors_without_black:
        color_string = array_to_string(color)
        if color_string not in map_color_index:  # if the plane is new
            map_color_index[color_string] = indx    # append (color:index) to map with the next index
            indx += 1

    unique_colors_without_black = filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)

    for i, color in enumerate(unique_colors_without_black):
        color_string = array_to_string(color)
        indices = np.where((point_colors == color).all(axis=1))
        plane_points = points[indices[0]]

        equations.append((get_normal(plane_points), map_color_index[color_string]))
    return equations