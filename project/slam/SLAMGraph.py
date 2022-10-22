from abc import abstractmethod
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

    @abstractmethod
    def _add_planes(self, pcd_s: List[Pcd], needed_indices: List[int]):
        pass

    @abstractmethod
    def _solve(self):
        pass

    def __build_graph(self, pcd_s: List[Pcd], needed_indices: List[int] = None):
        for _ in pcd_s:
            next_node = self.graph.add_node_pose_3d(geometry.SE3())
            self.graph_trajectory.append(next_node)

        self.graph.add_factor_1pose_3d(
            geometry.SE3(), self.graph_trajectory[0], 1e6 * np.identity(6)
        )

        for pcd in pcd_s:
            # add pcd_s len as we add plane nodes after view nodes
            real_indx = len(self.plane_index_to_real_index) + len(pcd_s)
            for plane in pcd.planes:
                if (
                        (needed_indices is None or plane.track in needed_indices)
                        and plane.track not in self.plane_index_to_real_index
                ):
                    self.plane_index_to_real_index[plane.track] = real_indx
                    real_indx += 1

    def estimate_graph(self, pcd_s: List[Pcd], needed_indices: List[int] = None):
        self.__build_graph(pcd_s, needed_indices)
        self._add_planes(pcd_s, needed_indices)
        self._solve()
        graph_estimated_state = self.graph.get_estimated_state()

        return graph_estimated_state
