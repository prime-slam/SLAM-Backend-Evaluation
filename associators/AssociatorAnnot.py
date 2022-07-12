
from Pcd import Pcd
from annotators.Annotator import Annotator


class AssociatorAnnot(Annotator):
    def __init__(self, pcd_s: list[Pcd], color_to_indx: dict):
        #super().__init__(pcd_s)
        super().__init__(pcd_s)
        self.__color_to_indx = color_to_indx


    def associate(self):
        if len(self.__color_to_indx) == 0:
            for pcd in self.__pcd_s:
                for plane in pcd.planes:  # pcd.planes
                    color_str = self.array_to_string(plane.color)
                    if color_str not in self.__color_to_indx:  # if the plane is new
                        plane.track = len(self.__color_to_indx) + 1
                        self.__color_to_indx[color_str] = plane.track  # append (color:index) to map with the next index

        return self.__pcd_s
