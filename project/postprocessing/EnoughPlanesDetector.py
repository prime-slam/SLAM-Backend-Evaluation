from typing import List

import numpy as np

from project.dto.Pcd import Pcd
from project.dto.Plane import Plane


class EnoughPlanesDetector:
    @staticmethod
    def has_enough_planes(pcd: Pcd) -> bool:
        return abs(EnoughPlanesDetector.__check_planes(pcd.planes)) > 0.1

    @staticmethod
    def __check_planes(planes: List[Plane]):
        matrix = []
        for plane in planes:
            matrix.append(plane.equation[:-1])
        matrix = np.asarray(matrix)
        covarience = matrix.T @ matrix
        eigvals, eigvects = np.linalg.eig(covarience)
        det = np.linalg.det(eigvects * eigvals)
        print("Det was {}".format(det))
        return det
