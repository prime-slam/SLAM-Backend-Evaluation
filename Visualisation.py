from typing import List

import numpy as np
import open3d as o3d

from Pcd import Pcd


class Visualisation(object):
    def __init__(self, graph_estimated_state, num_of_nodes: int):
        self.graph_estimated_state = graph_estimated_state
        self.num_of_nodes = num_of_nodes

    def visualisation(self, pcd_s: List[Pcd]):
        reflection = np.asarray(
            [[-1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]]
        )

        pc_answ = o3d.geometry.PointCloud()
        for i in range(self.num_of_nodes):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pcd_s[self.num_of_nodes - i - 1].points)

            build_color_matrix = np.zeros(np.shape(pcd_s[self.num_of_nodes - i - 1].points))
            for j, plane in enumerate(pcd_s[self.num_of_nodes - i - 1].planes):
                build_color_matrix[plane.plane_indices] = plane.color

            pc.colors = o3d.utility.Vector3dVector(build_color_matrix.astype(np.float64) / 255.0)
            pc = pc.transform(self.graph_estimated_state[-(i + 1)])
            pcd_s[self.num_of_nodes - i - 1].points = None
            pc = pc.voxel_down_sample(0.05)  # free memory immediately as vector3dvector copies data (don't know how to prevent it)
            pc = pc.transform(reflection)
            pc_answ += pc

        pc_answ = pc_answ.voxel_down_sample(0.05)

        o3d.visualization.draw_geometries([pc_answ])