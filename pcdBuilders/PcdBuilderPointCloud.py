import numpy as np
import open3d as o3d

from typing import List

from Visualisation import Visualisation
from annotators.AnnotatorPointCloud import AnnotatorPointCloud
from Camera import Camera
from dto.Pcd import Pcd
from pcdBuilders.PcdBuilder import PcdBuilder


class PcdBuilderPointCloud(PcdBuilder):
    def __init__(self, camera: Camera, annot: AnnotatorPointCloud):
        super().__init__(camera, annot)

    def _get_points(self, depth_image_path):
        pc = o3d.io.read_point_cloud(depth_image_path)

        return Pcd(np.asarray(pc.points) / 1000)
