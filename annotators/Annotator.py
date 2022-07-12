from abc import ABC, abstractmethod

from Pcd import Pcd


class Annotator(ABC):
    def __init__(self, array_path_annot: list[str]):
        self.array_path_to_annot = array_path_annot

    # @abstractmethod
    # def __get_planes(self, pcd: Pcd, image_colors: str):
    #     pass

    def annotate(self, pcd, image_number: int):
        pcd.planes = self.__get_planes(pcd, self.array_path_to_annot[image_number])

        return pcd

