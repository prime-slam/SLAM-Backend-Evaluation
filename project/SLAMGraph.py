from typing import List
from mrob.mrob import FGraph, geometry, LM
from dto.Pcd import Pcd

import numpy as np


class SLAMGraph:
    """
    Builds and estimtates the SLAM graph
    :attribute graph: object of a factor graph
    :attribute graph_trajectory: calculated trajectory of a system
    :attribute plane_index_to_real_index: map to match plane index and index used by the algorithm
    """

    def __init__(self):
        self.graph = FGraph()
        self.graph_trajectory = []
        self.plane_index_to_real_index = {}

    def __build_the_graph(self, pcd_s: List[Pcd], needed_indices: List[int]):
        for pcd in pcd_s:
            real_indx = len(self.plane_index_to_real_index)
            for plane in pcd.planes:
                if (
                    plane.track in needed_indices
                    and plane.track not in self.plane_index_to_real_index
                ):
                    self.plane_index_to_real_index[plane.track] = real_indx
                    real_indx += 1

        for _ in self.plane_index_to_real_index:
            self.graph.add_node_plane_4d(np.array([1, 0, 0, 0]))

        for _ in pcd_s:
            next_node = self.graph.add_node_pose_3d(geometry.SE3())
            self.graph_trajectory.append(next_node)
        self.graph.add_factor_1pose_3d(
            geometry.SE3(), self.graph_trajectory[0], 1e6 * np.identity(6)
        )

    def estimate_the_graph(self, pcd_s: list[Pcd], needed_indices: List[int]):

        self.__build_the_graph(pcd_s, needed_indices)
        w_z = np.identity(4)  # weight matrix

        for i, pcd in enumerate(pcd_s):
            for plane in pcd.planes:
                if plane.track in needed_indices:
                    cur_indx = self.plane_index_to_real_index[plane.track]
                    self.graph.add_factor_1pose_1plane_4d(
                        plane.equation, self.graph_trajectory[i], cur_indx, w_z
                    )

        self.graph.solve(LM)
        graph_estimated_state = self.graph.get_estimated_state()

        return graph_estimated_state
