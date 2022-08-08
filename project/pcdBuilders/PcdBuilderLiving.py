import cv2
import numpy as np

from project.annotators.AnnotatorImage import AnnotatorImage
from project.Camera import Camera
from project.dto.Pcd import Pcd
from project.pcdBuilders.PcdBuilder import PcdBuilder


class PcdBuilderLiving(PcdBuilder):
    """
    Builds pcd from depth image, uses pinhole camera parameters
    """

    def __init__(self, camera: Camera, annot: AnnotatorImage):
        super().__init__(camera, annot)

    def __convert_from_plane_to_3d(self, u, v, depth):
        x_over_z = (
            v - self.cam.cx
        ) / self.cam.focal_x  # создать матрицу result (rows, colums, 3)
        y_over_z = (u - self.cam.cy) / self.cam.focal_y

        z_matrix = depth / self.cam.scale

        x_matrix = x_over_z * z_matrix
        y_matrix = y_over_z * z_matrix

        return np.dstack((x_matrix, y_matrix, z_matrix))

    def _get_points(self, depth_image_path) -> Pcd:

        matrix_depth = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

        rows, columns = matrix_depth.shape
        columns_indices = np.arange(columns)

        matrix_v = np.tile(columns_indices, (rows, 1))
        matrix_u = np.transpose(np.tile(np.arange(rows), (columns, 1)))

        matrix_xyz = self.__convert_from_plane_to_3d(
            matrix_u,
            matrix_v,
            matrix_depth,
        )
        return Pcd(matrix_xyz.reshape(-1, matrix_xyz.shape[2]))
