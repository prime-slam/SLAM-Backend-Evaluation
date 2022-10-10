from typing import List

from project.associators.Associator import Associator
from project.dto.Pcd import Pcd


class PlaneRemover:
    @staticmethod
    def remove_by_colors(pcd: Pcd, color_strs: List[str]) -> Pcd:
        result = Pcd(pcd.points)
        for plane in pcd.planes:
            if plane.color is None or Associator.make_string_from_array(plane.color) not in color_strs:
                result.planes.append(plane)

        return result
