from typing import List, Dict, Any

import numpy as np
from mrob.mrob import FGraph, geometry, registration, LM

from Pcd import Pcd


class SLAMGraph(object):

    def __init__(self):
        self.graph = FGraph()
        self.graph_trajectory = []
        self.plane_index_to_real_index = {}

    def __build_the_graph(self, pcd_s: List[Pcd], num_of_nodes: int, first_node: int, needed_indices: List[int]):
        needed_pcds = pcd_s[first_node: first_node + num_of_nodes]
        for pcd in needed_pcds:
            real_indx = len(self.plane_index_to_real_index)
            for plane in pcd.planes:
                if plane.track in needed_indices:
                    if plane.track not in self.plane_index_to_real_index:
                        self.plane_index_to_real_index[plane.track] = real_indx
                        real_indx += 1

        for i, _ in enumerate(self.plane_index_to_real_index):
            self.graph.add_node_plane_4d(
                np.array([1, 0, 0, 0]))

        for i in range(num_of_nodes):
            next_node = self.graph.add_node_pose_3d(geometry.SE3())
            self.graph_trajectory.append(next_node)
        self.graph.add_factor_1pose_3d(geometry.SE3(), self.graph_trajectory[0], 1e6 * np.identity(6))
        return needed_pcds

    def estimate_the_graph(self, pcd_s: list[Pcd], num_of_nodes: int, first_node: int, needed_indices: List[int]):

        needed_pcds = self.__build_the_graph(pcd_s, num_of_nodes, first_node, needed_indices)
        w_z = np.identity(4)  # weight matrix

        for n, pcd in enumerate(needed_pcds):
            for plane in pcd.planes:
                if plane.track in needed_indices:
                    cur_indx = self.plane_index_to_real_index[plane.track]
                    self.graph.add_factor_1pose_1plane_4d(plane.equation, self.graph_trajectory[n], cur_indx, w_z)

        self.graph.solve(LM)
        graph_estimated_state = self.graph.get_estimated_state()

        return graph_estimated_state