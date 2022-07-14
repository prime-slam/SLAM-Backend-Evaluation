from typing import List

import read_office

from Camera import Camera
from Pcd import Pcd
from annotators.AnnotaatorImage import AnnotatorImage
from pcdBuilders.PcdBuilder import PcdBuilder


class PcdBuilderOffice(PcdBuilder):
    def __init__(self, camera: Camera, annot_path: List[str]):
        super().__init__(camera)
        self.annot = AnnotatorImage(annot_path)

    def _get_points(self, i: int, array_file_names: List[str]):
        points_of_image = read_office.getting_points(i, array_file_names, self.cam)
        return Pcd(points_of_image.reshape(-1, points_of_image.shape[2]))

