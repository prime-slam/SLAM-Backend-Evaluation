import numpy as np
import open3d as o3d

from project.annotators.AnnotatorPointCloud import AnnotatorPointCloud
from project.Camera import Camera
from project.dto.Pcd import Pcd
from project.pcdBuilders.PcdBuilder import PcdBuilder


class PcdBuilderPointCloud(PcdBuilder):
    """
    Builds pcd from .pcd and .npy files
    """

    def __init__(self, camera: Camera, annot: AnnotatorPointCloud):
        super().__init__(camera, annot)

    def _get_points(self, depth_image_path):
        pc = o3d.io.read_point_cloud(depth_image_path)
        to_meters = 1000

        return Pcd(np.asarray(pc.points) / to_meters)
