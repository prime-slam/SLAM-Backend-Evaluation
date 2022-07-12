import cv2
import numpy as np

from annotators.AnnotaatorImage import AnnotatorImage
from Camera import Camera
from Pcd import Pcd
from pcdBuilders.PcdBuilder import PcdBuilder


class PcdBuilderLiving(PcdBuilder):
    def __init__(self, camera: Camera, annot_path: list[str]):
        super().__init__(camera)
        self.annot = AnnotatorImage(annot_path)

    def __convert_from_plane_to_3d(self, u, v, depth):
        x_over_z = (v - self.cam.cx) / self.cam.focal_x  # создать матрицу result (rows, colums, 3)
        y_over_z = (u - self.cam.cy) / self.cam.focal_y

        z_matrix = depth / self.cam.scale

        x_matrix = x_over_z * z_matrix
        y_matrix = y_over_z * z_matrix

        return np.dstack((x_matrix, y_matrix, z_matrix))

    def __get_points(self, i: int, array_file_names: list[str]) -> Pcd:

        matrix_depth = cv2.imread(array_file_names[i], cv2.IMREAD_ANYDEPTH)

        rows, columns, _ = matrix_depth.shape
        columns_indices = np.arange(columns)

        matrix_v = np.tile(columns_indices, (rows, 1))
        matrix_u = np.transpose(np.tile(np.arange(rows), (columns, 1)))

        matrix_xyz = self.__convert_from_plane_to_3d(
            matrix_u,
            matrix_v,
            matrix_depth,
        )
        return Pcd(matrix_xyz.reshape(-1, matrix_xyz.shape[2]))

    def build_pcd(self, image_number: int, array_file_names_depth: list[str]):
        pcd = self.__get_points(image_number, array_file_names_depth)
        pcd = self.annot.annotate(pcd, image_number)

        return pcd
