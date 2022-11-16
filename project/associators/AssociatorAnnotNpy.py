from typing import List
from project.dto.Pcd import Pcd
from project.associators.Associator import Associator
from project.utils.colors import get_random_color, color_to_string, color_from_string


class AssociatorAnnotNpy(Associator):
    """
    Associates planes with annotation
    :attribute __color_to_indx: map to align color of a plane to its index
    """

    def __init__(self):
        self.__track_to_color = {}

    def associate(self, pcd_s: List[Pcd]):
        for pcd in pcd_s:
            for plane in pcd.planes:
                if plane.track not in self.__track_to_color:
                    self.__track_to_color[plane.track] = color_to_string(
                        get_random_color()
                    )
                plane.color = color_from_string(self.__track_to_color[plane.track])

        return pcd_s
