import numpy as np
import math
import cv2
import open3d as o3d


class Camera(object):
    def __init__(self, width, height, cx, cy, focal_x, focal_y, scale):
        self.width = width
        self.height = height
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


def string_to_array(string):
    values = string.split('#')
    answ = []

    for value in values:
        answ.append(int(value))

    return np.asarray(answ)

def convert_from_plane_to_3d(u, v, depth, cam: Camera):
    #result = np.zeros((u.shape, 3))
    x_over_z = (v - cam.cx) / cam.focal_x # создать матрицу result (rows, colums, 3)
    y_over_z = (u - cam.cy) / cam.focal_y

    # result[::2] = depth / cam.scale
    # result[::1] = y_over_z * result[::2]
    # result[::0] = x_over_z * result[::2]
    z_matrix = depth / cam.scale

    x_matrix = x_over_z * z_matrix
    y_matrix = y_over_z * z_matrix

    return np.dstack((x_matrix, y_matrix, z_matrix))


def generate_color(set_of_generated_colors):
    answ = np.random.randint(255, size=3)
    while array_to_string(answ) in set_of_generated_colors:
        answ = generate_color(set_of_generated_colors)
    set_of_generated_colors.add(array_to_string(answ))
    return answ


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


def building_maps(points, colors, color_to_index, indx_to_max_num_points):

    colors_unique = np.unique(colors, axis=0)

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
        indices = np.where((colors == color).all(axis=1))
        if (cur_indx in indx_to_max_num_points and len(indices[0]) > indx_to_max_num_points[cur_indx]) \
                or cur_indx not in indx_to_max_num_points:
            indx_to_max_num_points[cur_indx] = len(indices[0])

    return color_to_index, indx_to_max_num_points


def equation_extraction(points, colors, color_to_index, max_planes, planes):
    colors_unique = np.unique(colors, axis=0)

    unique_colors_without_black = list(filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique))
    planes_of_image = []
    cur_colors = np.zeros(points.shape)
    for color in unique_colors_without_black:
        color_string = array_to_string(color)
        cur_indx = color_to_index[color_string]
        indices = np.where((colors == color).all(axis=1))
        if cur_indx in max_planes:
            plane_points = points[indices[0]]
            cur_colors[indices] = colors[indices]
            plane = Plane(get_normal(plane_points), color_to_index[color_string])
            planes_of_image.append(plane)
    planes.append(planes_of_image)
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(points)
    # pc.colors = o3d.utility.Vector3dVector(cur_colors.astype(np.float64) / 255.0)
    # o3d.visualization.draw_geometries([pc])

    return planes


def building_maps_frontend(points, annot, indx_to_color, indx_to_max_num_points):

    annot_unique = np.unique(annot, axis=0)

    num_unique_colors = annot_unique.size
    num_of_points = points.size

    unique_indices_without_black = list(filter(lambda x: (x != 1), annot_unique))

    for color_index in unique_indices_without_black:  # if the plane is new
        indx_to_color[color_index] = array_to_string(generate_color())  # append (color:index) to map with the next index
        indices = np.where((annot == color_index))
        if (color_index in indx_to_max_num_points and len(indices[0]) > indx_to_max_num_points[color_index]) \
                or color_index not in indx_to_max_num_points:
            indx_to_max_num_points[color_index] = len(indices[0])

    return indx_to_color, indx_to_max_num_points


def planes_extraction_for_frontend(points, annot, indx, indx_to_max_num_points, prev_planes, indx_to_color,
                                   set_of_generated_colors):

    planes_2_matched = []
    good_ones = []
    true_colors = np.zeros(points.shape)

    annot_unique = np.unique(annot, axis=0)

    # нужен глобальный индекс
    unique_annot_without_black = list(filter(lambda x: (x != 1), annot_unique))

    indices_of_planes = {}
    for i, annot_num in enumerate(unique_annot_without_black):

        indx = len(indx_to_max_num_points)
        indices = np.where(annot == annot_num)
        plane_points = points[indices[0]]
        equation = get_normal(plane_points)
        #len_before_loop = len(planes_2_matched)
        cur_indx = 0
        if prev_planes != None:
            min_bias = 10
            most_correct_plane = -1
            for plane in prev_planes:
                deviation = np.dot(plane.equation[:3], equation[:3]) # косинус угла: наибольший, когда плоскости совпадают
                bias = math.acos(deviation) + math.fabs(plane.equation[-1] - equation[-1])
                if bias < min_bias:
                    min_bias = bias
                    most_correct_plane = plane.index
            if most_correct_plane == -1:
                #indx += 1
                p = Plane(equation, indx)
                indx_to_color[indx] = generate_color(set_of_generated_colors)
                true_colors[indices[0]] = indx_to_color[indx]
                planes_2_matched.append(p)
                indices_of_planes[indx] = len(indices[0])
            elif most_correct_plane != -1:
                if min_bias < 1 and most_correct_plane not in indices_of_planes:
                    indices_of_planes[most_correct_plane] = len(indices[0])
                    p = Plane(equation, most_correct_plane)
                    planes_2_matched.append(p)
                    if (most_correct_plane in indx_to_max_num_points and len(indices[0]) > indx_to_max_num_points[most_correct_plane]) \
                            or most_correct_plane not in indx_to_max_num_points:
                        indx_to_max_num_points[most_correct_plane] = len(indices[0])
                    true_colors[indices[0]] = indx_to_color[most_correct_plane]
                elif min_bias < 1 and most_correct_plane in indices_of_planes:
                    if (len(indices[0]) > indices_of_planes[most_correct_plane]):
                        for pl in planes_2_matched:
                            if pl.index == most_correct_plane:
                                planes_2_matched.remove(pl)
                                break
                        p = Plane(equation, most_correct_plane)
                        planes_2_matched.append(p)
                        if (most_correct_plane in indx_to_max_num_points and len(indices[0]) > indx_to_max_num_points[
                            most_correct_plane]) \
                                or most_correct_plane not in indx_to_max_num_points:
                            indx_to_max_num_points[most_correct_plane] = len(indices[0])
                        indices_of_planes[most_correct_plane] = len(indices[0])
                        true_colors[indices[0]] = indx_to_color[most_correct_plane]
        else:
            p = Plane(equation, indx)
            indx_to_color[indx] = generate_color(set_of_generated_colors)
            true_colors[indices[0]] = indx_to_color[indx]
            indices_of_planes[indx] = len(indices[0])
            planes_2_matched.append(p)
            if (indx in indx_to_max_num_points and len(indices[0]) > indx_to_max_num_points[indx]) \
                    or indx not in indx_to_max_num_points:
                indx_to_max_num_points[indx] = len(indices[0])
    return indx, planes_2_matched, true_colors






