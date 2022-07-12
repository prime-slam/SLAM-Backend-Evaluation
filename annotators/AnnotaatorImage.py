import cv2
import numpy as np


from Plane import Plane
from annotators.Annotator import Annotator


class AnnotatorImage(Annotator):
    def __init__(self, array_path_annot: list[str]):
        super().__init__(array_path_annot)

    def __get_planes(self, pcd, image_colors: str):
        planes_of_image = []

        matrix_colors = cv2.imread(image_colors, cv2.IMREAD_COLOR)
        colors_reshaped = matrix_colors.reshape(-1, matrix_colors.shape[2])
        colors_unique = np.unique(colors_reshaped, axis=0)

        unique_colors_without_black = filter(lambda x: (x != [0, 0, 0]).all(axis=0), colors_unique)
        for color in unique_colors_without_black:
            indices = np.where((colors_reshaped == color).all(axis=1))[0]
            plane_points = pcd.points[indices]
            plane = Plane(Plane.get_normal(plane_points), track=-1, color=color, indices=indices)
            planes_of_image.append((plane))
        return planes_of_image