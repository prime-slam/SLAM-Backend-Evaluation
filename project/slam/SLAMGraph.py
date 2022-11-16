from abc import abstractmethod
from typing import List

from mrob import mrob
from mrob.mrob import FGraph, geometry, LM

import numpy as np

from project.dto.Pcd import Pcd


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
    def _add_plane_node(self) -> int:
        pass

    @abstractmethod
    def _solve(self):
        pass

    def __build_graph(
        self, pcd_s: List[Pcd], needed_indices: List[int] = None, initial_poses=None
    ):
        for index, _ in enumerate(pcd_s):
            item = (
                geometry.SE3()
                if initial_poses is None
                else geometry.SE3(initial_poses[index])
            )
            if index == 0:
                next_node = self.graph.add_node_pose_3d(item, mrob.NODE_ANCHOR)
            else:
                next_node = self.graph.add_node_pose_3d(item)
            self.graph_trajectory.append(next_node)

        for pcd in pcd_s:
            for plane in pcd.planes:
                if (
                    needed_indices is None or plane.track in needed_indices
                ) and plane.track not in self.plane_index_to_real_index:
                    real_index = self._add_plane_node()
                    self.plane_index_to_real_index[plane.track] = real_index

    def estimate_graph(
        self, pcd_s: List[Pcd], needed_indices: List[int] = None, initial_poses=None
    ):
        self.__build_graph(pcd_s, needed_indices, initial_poses)
        self._add_planes(pcd_s, needed_indices)
        self._solve()
        graph_estimated_state = self.graph.get_estimated_state()

        return graph_estimated_state
