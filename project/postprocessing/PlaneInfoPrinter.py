from typing import List

from project.dto.Pcd import Pcd
from project.utils.colors import color_to_string


class PlaneInfoPrinter:
    @staticmethod
    def print_planes_info(pcds: List[Pcd]):
        data = {}
        for pcd in pcds:
            for plane in pcd.planes:
                color_str = color_to_string(plane.color)
                if color_str not in data:
                    data[color_str] = "id: {0}, pts_cnt: {1}".format(plane.track, len(plane.plane_indices))

        print(data)
