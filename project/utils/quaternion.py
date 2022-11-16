from typing import List

import numpy as np
from scipy.spatial.transform import Rotation


def read_poses_csv(path_to_file: str) -> List[np.array]:
    result = []
    with open(path_to_file, 'r') as poses_file:
        for line in poses_file:
            values = line.split(",")
            x, y, z = list(map(lambda x: float(x), values[:3]))
            qx, qy, qz, qw = list(map(lambda x: float(x), values[3:7]))
            rotation_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            transform_matrix = np.zeros((4, 4), dtype=float)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = np.asarray([x, y, z])
            transform_matrix[3, 3] = 1
            result.append(transform_matrix)

    return result
