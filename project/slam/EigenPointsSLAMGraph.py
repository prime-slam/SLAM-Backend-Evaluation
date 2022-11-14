from typing import List

from project.dto.Pcd import Pcd
from project.slam.SLAMGraph import SLAMGraph
from mrob.mrob import LM_ELLIPS


class EigenPointsSLAMGraph(SLAMGraph):

    def _add_plane_node(self) -> int:
        return self.graph.add_eigen_factor_plane()

    def _add_planes(self, pcd_s: List[Pcd], needed_indices: List[int]):
        for i, pcd in enumerate(pcd_s):
            for plane in pcd.planes:
                if needed_indices is not None and plane.track not in needed_indices:
                    continue

                cur_indx = self.plane_index_to_real_index[plane.track]
                plane_points = pcd.points[plane.plane_indices]
                self.graph.eigen_factor_plane_add_points_array(
                    planeEigenId=cur_indx,
                    nodePoseId=self.graph_trajectory[i],
                    pointsArray=plane_points,
                    W=1.0
                )

    def _solve(self):
        self.graph.solve(LM_ELLIPS)
