import numpy as np
import open3d as o3d

from project.annotators.AnnotatorPointCloud import AnnotatorPointCloud
from project.Camera import Camera
from project.dto.Pcd import Pcd
from project.pcdBuilders.PcdBuilder import PcdBuilder


class PcdBuilderKitti(PcdBuilder):
    """
    Builds pcd from .bin kitti files
    """

    def __init__(
        self,
        camera: Camera,
        annot: AnnotatorPointCloud,
        reflection=None,
    ):
        super().__init__(camera, annot)
        self.reflection = reflection

    def _get_points(self, depth_image_path):
        pcd_points = np.fromfile(depth_image_path, dtype=np.float32).reshape(-1, 4)
        pc = o3d.geometry.PointCloud()
        # data contains [x, y, z, reflectance] for each point -- we skip the last one
        pc.points = o3d.utility.Vector3dVector(pcd_points[:, :3])
        if self.reflection is not None:
            pc = pc.transform(self.reflection)

        return Pcd(np.asarray(pc.points))
