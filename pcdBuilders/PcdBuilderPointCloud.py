import numpy as np

from annotators.AnnotatorPointCloud import AnnotatorPointCloud
from Camera import Camera
from Pcd import Pcd


import open3d as o3d

from pcdBuilders.PcdBuilder import PcdBuilder


class PcdBuilderPointcloud(PcdBuilder):
    def __init__(self, camera: Camera, annot_path: list[str]):
        super().__init__(camera)
        self.annot = AnnotatorPointCloud(annot_path)

    def __get_points(self, image_number: int, array_file_names: list[str]):
        pc = o3d.io.read_point_cloud(array_file_names[image_number])

        return Pcd(np.asarray(pc.points) / 1000)
