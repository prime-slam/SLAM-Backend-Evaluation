import os.path
from typing import List
from project.dto.Plane import Plane
from project.annotators.Annotator import Annotator

import cv2
import numpy as np


class AnnotatorImage(Annotator):
    """
    Extracts planes from annotation in rgb format
    """

    def __init__(
        self,
        array_path_annot: List[str],
        array_path_depth: List[str],
        is_office: bool = False,
    ):
        super().__init__(array_path_annot)
        self.depth_to_rgb = AnnotatorImage.__match_rgb_with_depth(
            array_path_annot, array_path_depth, is_office
        )

    def _get_planes(self, pcd, image_colors: str):
        planes_of_image = []

        matrix_colors = cv2.imread(image_colors, cv2.IMREAD_COLOR)
        colors_reshaped = matrix_colors.reshape(-1, matrix_colors.shape[2])
        colors_unique = np.unique(colors_reshaped, axis=0)

        unique_colors_without_black = list(
            filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)
        )
        for color in unique_colors_without_black:
            indices = np.where((colors_reshaped == color).all(axis=1))[0]
            plane_points = pcd.points[indices]
            equation = Plane.get_equation(plane_points)
            plane = Plane(equation, track=-1, color=color, indices=indices)
            planes_of_image.append((plane))

        return planes_of_image

    def _get_annot_image_number_by_depth(self, depth_image_number: int):
        return self.depth_to_rgb[depth_image_number]

    @staticmethod
    def __match_rgb_with_depth(rgb_filenames, depth_filenames, is_office: bool) -> list:
        rgb_filenames = list(map(lambda x: os.path.split(x)[-1], rgb_filenames))
        depth_filenames = list(map(lambda x: os.path.split(x)[-1], depth_filenames))
        depth_to_rgb_index = []
        rgb_index = 0
        depth_index = 0
        prev_delta = float("inf")
        while depth_index < len(depth_filenames) and rgb_index < len(rgb_filenames):
            if is_office:
                rgb_timestamp = float(rgb_filenames[rgb_index][-8:-4])
                depth_timestamp = float(depth_filenames[depth_index][-10:-6])
            else:
                rgb_timestamp = float(rgb_filenames[rgb_index][:-4])
                depth_timestamp = float(depth_filenames[depth_index][:-4])
            delta = abs(depth_timestamp - rgb_timestamp)

            if rgb_timestamp < depth_timestamp:
                prev_delta = delta
                rgb_index += 1
                continue

            if prev_delta < delta:
                depth_to_rgb_index.append(rgb_index - 1)
            else:
                depth_to_rgb_index.append(rgb_index)

            depth_index += 1

        # Fix case when the last timestamp was for depth img
        while depth_index < len(depth_filenames):
            depth_to_rgb_index.append(rgb_index - 1)
            depth_index += 1

        return depth_to_rgb_index
