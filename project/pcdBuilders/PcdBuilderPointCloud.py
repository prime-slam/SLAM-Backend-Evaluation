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

    def __init__(
        self,
        camera: Camera,
        annot: AnnotatorPointCloud,
        reflection=None,
        scale=1000
    ):
        super().__init__(camera, annot)
        self.reflection = reflection
        self.scale = scale

    def _get_points(self, depth_image_path):
        pc = o3d.io.read_point_cloud(depth_image_path)
        if self.reflection is not None:
            pc = pc.transform(self.reflection)
        to_meters = self.scale

        return Pcd(np.asarray(pc.points) / to_meters)
