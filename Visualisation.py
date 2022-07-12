import numpy as np
import open3d as o3d

from Pcd import Pcd


class Visualisation(object):
    def __init__(self, graph_estimated_state, num_of_nodes: int, pcd_s: list[Pcd]):
        self.graph_estimated_state = graph_estimated_state
        self.num_of_nodes = num_of_nodes
        self.pcd_s = pcd_s

    def visualisation(self):
        reflection = np.asarray(
            [[-1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]]
        )

        pc_answ = o3d.geometry.PointCloud()
        for i in range(self.num_of_nodes):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(self.pcd_s[self.num_of_nodes - i - 1])

            build_color_matrix = np.zeros(np.size(self.pcd_s[i].points))
            for plane in self.pcd_s[i].planes:
                build_color_matrix[plane.indices] = plane.color

            pc.colors = o3d.utility.Vector3dVector(build_color_matrix.astype(np.float64) / 255.0)
            pc = pc.transform(self.graph_estimated_state[-(i + 1)])
            self.pcd_s[self.num_of_nodes - i - 1].points = None
            pc = pc.voxel_down_sample(
                0.05)  # free memory immediately as vector3dvector copies data (don't know how to prevent it)
            pc = pc.transform(reflection)
            pc_answ += pc

        pc_answ = pc_answ.voxel_down_sample(0.05)

        o3d.visualization.draw_geometries([pc_answ])