from typing import List
from dto.Pcd import Pcd

import numpy as np
import open3d as o3d

from project.utils.colors import denormalize_color


class Visualisation:
    def __init__(self, graph_estimated_state):
        self.graph_estimated_state = graph_estimated_state

    @staticmethod
    def visualize_pcd(pcd: Pcd, transforms=None) -> o3d.geometry.PointCloud:
        if transforms is None:
            transforms = []

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pcd.points)

        build_color_matrix = np.zeros(np.shape(pcd.points))
        for j, plane in enumerate(pcd.planes):
            build_color_matrix[plane.plane_indices] = plane.color

        transforms.append(
            np.asarray([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        )

        pc.colors = o3d.utility.Vector3dVector(
            build_color_matrix.astype(np.float64) / 255.0
        )
        for transform in transforms:
            pc = pc.transform(transform)
        return pc

    def visualize(self, pcd_s: List[Pcd], graph_estimated_state):
        num_of_nodes = len(pcd_s)
        reflection = np.asarray(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        pc_answ = o3d.geometry.PointCloud()
        for i in range(num_of_nodes):
            pc = self.visualize_pcd(
                pcd_s[-(i + 1)], [graph_estimated_state[-(i + 1)], reflection]
            )
            pc = pc.voxel_down_sample(0.05)

            pcd_s[-(i + 1)].points = None
            pc_answ += pc
        pc_answ = pc_answ.voxel_down_sample(0.05)

        Visualisation.__draw_pcd(pc_answ)

    @staticmethod
    def __draw_pcd(pcd: o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()

        picked_points = vis.get_picked_points()
        for point in picked_points:
            print(
                "Pont with position {0} picked. Color: {1}".format(
                    points[point],
                    denormalize_color(colors[point])
                )
            )

