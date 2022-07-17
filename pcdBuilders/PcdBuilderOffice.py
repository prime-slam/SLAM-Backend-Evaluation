from typing import List

import read_office

from Camera import Camera
from dto.Pcd import Pcd
from annotators.AnnotatorImage import AnnotatorImage
from pcdBuilders.PcdBuilder import PcdBuilder


class PcdBuilderOffice(PcdBuilder):
    def __init__(self, camera: Camera, annot: AnnotatorImage):
        super().__init__(camera, annot)

    def _get_points(self, depth_image_path):
        points_of_image = read_office.getting_points(depth_image_path, self.cam)
        return Pcd(points_of_image.reshape(-1, points_of_image.shape[2]))
