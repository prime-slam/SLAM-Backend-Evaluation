from typing import List

from project import config
from project.dto.Pcd import Pcd


class PostProcessing:
    """
    Chooses planes with maximum points
    """

    @staticmethod
    def post_process(pcd_s: List[Pcd], verbose=False):
        """
        :param pcd_s: list of pcd objects
        :return: indices of planes with maximum points
        """
        indx_to_max_num_points = {}
        for i, pcd in enumerate(pcd_s):
            if verbose:
                print("postprocessing " + str(i))
            for plane in pcd.planes:
                num_of_points = len(plane.plane_indices)
                if (
                    plane.track not in indx_to_max_num_points
                    or num_of_points > indx_to_max_num_points[plane.track]
                ):
                    indx_to_max_num_points[plane.track] = num_of_points
        map_indx_to_max_num_points = sorted(
            indx_to_max_num_points, key=indx_to_max_num_points.get
        )
        return map_indx_to_max_num_points[-config.MAX_PLANES_COUNT:]
