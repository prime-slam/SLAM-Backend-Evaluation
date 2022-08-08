from typing import List
from scripts.dto.Pcd import Pcd
from scripts.associators.Associator import Associator


class AssociatorAnnot(Associator):
    """
    Associates planes with annotation
    :attribute __color_to_indx: map to allign color of a plane to its index
    """

    def __init__(self, color_to_indx={}):
        self.__color_to_indx = color_to_indx

    def associate(self, pcd_s: List[Pcd]):
        for i, pcd in enumerate(pcd_s):
            for plane in pcd.planes:
                color_str = self.array_to_string(plane.color)
                if color_str not in self.__color_to_indx:  # if the plane is new
                    plane.track = len(self.__color_to_indx) + 1
                    self.__color_to_indx[
                        color_str
                    ] = plane.track  # append (color:index) to map with the next index
                else:
                    plane.track = self.__color_to_indx[color_str]

        return pcd_s
