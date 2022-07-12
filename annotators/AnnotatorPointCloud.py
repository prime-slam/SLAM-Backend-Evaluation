import numpy as np


from Plane import Plane
from annotators.Annotator import Annotator


class AnnotatorPointCloud(Annotator):
    def __init__(self, array_path_annot: list[str]):
        super().__init__(array_path_annot)

    def __get_planes(self, pcd, image_colors: str):
        planes = []
        annot_of_image = np.load(image_colors)
        annot_unique = np.unique(annot_of_image, axis=0)
        unique_annot_without_black = list(filter(lambda x: (x != 1), annot_unique))

        for i, annot_num in enumerate(unique_annot_without_black):
            indices = np.where(annot_of_image == annot_num)[0]
            plane_points = pcd.points[indices]
            equation = Plane.get_normal(plane_points)
            plane = Plane(equation, track=-1, color=np.asarray([0, 0, 0]), indices=indices)
            planes.append(plane)

        return planes
