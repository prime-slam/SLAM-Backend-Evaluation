from typing import List
from mrob.mrob import FGraph, LM

import numpy as np

from project.dto.Pcd import Pcd
from project.slam.SLAMGraph import SLAMGraph


class LandmarksSLAMGraph(SLAMGraph):
    def _add_plane_node(self) -> int:
        return self.graph.add_node_plane_4d(np.array([1, 0, 0, 0]))

    def _add_planes(self, pcd_s: List[Pcd], needed_indices: List[int]):
        w_z = np.identity(4)  # weight matrix
        for i, pcd in enumerate(pcd_s):
            for plane in pcd.planes:
                if needed_indices is not None and plane.track not in needed_indices:
                    continue

                cur_indx = self.plane_index_to_real_index[plane.track]
                self.graph.add_factor_1pose_1plane_4d(
                    plane.equation, self.graph_trajectory[i], cur_indx, w_z
                )

    def _solve(self):
        self.graph.solve(LM)
