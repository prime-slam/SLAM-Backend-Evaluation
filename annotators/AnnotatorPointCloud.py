import sys
from typing import List
from dto.Plane import Plane
from annotators.Annotator import Annotator

import numpy as np


class AnnotatorPointCloud(Annotator):
    def __init__(self, array_path_annot: List[str]):
        super().__init__(array_path_annot)

    def _get_planes(self, pcd, image_colors: str):
        planes = []
        equations = []
        annot_of_image = np.load(image_colors)
        annot_unique = np.unique(annot_of_image, axis=0)
        unique_annot_without_black = list(filter(lambda x: (x != 1), annot_unique))

        for i, annot_num in enumerate(unique_annot_without_black):
            indices = np.where(annot_of_image == annot_num)[0]

            plane_points = pcd.points[indices]

            equation = Plane.get_normal(plane_points)
            equations.append(equation)

            plane = Plane(
                equation, track=-1, color=np.asarray([0, 0, 0]), indices=indices
            )
            planes.append(plane)

        return planes
