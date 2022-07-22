import math
import os

import numpy as np


def __filenames_sorted_mapper(filename: str) -> int:
    return int(filename.split(".")[0].split("_")[-1])


def __load_camera_params_from_file(depth_image) -> dict:
    result = {}
    params_path = depth_image[:-5] + "txt"
    with open(params_path, "r") as input_file:
        for line in input_file:
            field_name_start = 0
            field_name_end = line.find(" ")
            field_name = line[field_name_start:field_name_end]
            value_start = line.find("=") + 2  # skip space after '='
            if field_name == "cam_angle":
                value_end = line.find(";")
            else:
                value_end = line.find(";") - 1
            value = line[value_start:value_end]
            result[field_name] = value

        return result


def provide_filenames(general_path) -> (list, list):
    path = general_path  # as paths are equal
    filenames = os.listdir(path)
    rgb_filenames = filter(lambda x: x.endswith(".png"), filenames)
    depth_filenames = filter(lambda x: x.endswith(".depth"), filenames)

    rgb_filenames = sorted(rgb_filenames, key=__filenames_sorted_mapper)
    depth_filenames = sorted(depth_filenames, key=__filenames_sorted_mapper)

    full_rgb_filenames = []
    full_depth_filenames = []

    for rgb_filename in rgb_filenames:
        full_rgb_filenames.append(os.path.join(general_path, rgb_filename))

    for depth_filename in depth_filenames:
        full_depth_filenames.append(os.path.join(general_path, depth_filename))

    return full_rgb_filenames, full_depth_filenames


def __get_camera_params_for_frame(depth_image):
    # Adopted from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/getcamK.m
    camera_params_raw = __load_camera_params_from_file(depth_image)
    cam_dir = np.fromstring(camera_params_raw["cam_dir"][1:-1], dtype=float, sep=",").T
    cam_right = np.fromstring(
        camera_params_raw["cam_right"][1:-1], dtype=float, sep=","
    ).T
    cam_up = np.fromstring(camera_params_raw["cam_up"][1:-1], dtype=float, sep=",").T
    focal = np.linalg.norm(cam_dir)
    aspect = np.linalg.norm(cam_right) / np.linalg.norm(cam_up)
    angle = 2 * math.atan(np.linalg.norm(cam_right) / 2 / focal)

    width = 640
    height = 480
    psx = 2 * focal * math.tan(0.5 * angle) / width
    psy = 2 * focal * math.tan(0.5 * angle) / aspect / height

    psx = psx / focal
    psy = psy / focal

    o_x = (width + 1) * 0.5
    o_y = (height + 1) * 0.5

    fx = 1 / psx
    fy = -1 / psy
    cx = o_x
    cy = o_y

    return fx, fy, cx, cy


def getting_points(depth_frame_path, cam_intrinsic):
    # Adopted from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/compute3Dpositions.m
    fx, fy, cx, cy = (
        cam_intrinsic.focal_x,
        cam_intrinsic.focal_y,
        cam_intrinsic.cx,
        cam_intrinsic.cy,
    )

    x_matrix = np.tile(
        np.arange(cam_intrinsic.width), (cam_intrinsic.height, 1)
    ).flatten()
    y_matrix = np.transpose(
        np.tile(np.arange(cam_intrinsic.height), (cam_intrinsic.width, 1))
    ).flatten()
    x_modifier = (x_matrix - cx) / fx
    y_modifier = (y_matrix - cy) / fy

    points = np.zeros((cam_intrinsic.width * cam_intrinsic.height, 3))

    with open(depth_frame_path, "r") as input_file:
        data = input_file.read()
        depth_data = np.asarray(
            list(
                map(
                    lambda x: float(x),
                    data.split(" ")[: cam_intrinsic.height * cam_intrinsic.width],
                )
            )
        )
        # depth_data = depth_data.reshape((480, 640))

        scale = 100  # from cm to m
        points[:, 2] = (
            depth_data / np.sqrt(x_modifier**2 + y_modifier**2 + 1) / scale
        )
        points[:, 0] = x_modifier * points[:, 2]
        points[:, 1] = y_modifier * points[:, 2]

    return points
