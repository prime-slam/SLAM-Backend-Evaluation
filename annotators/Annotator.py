from abc import ABC, abstractmethod
from typing import List

from dto.Pcd import Pcd
from dto.Plane import Plane


class Annotator(ABC):
    """
    Extracts planes from annotation
    :attribute array_path_to_annot: path to annotated images
    """

    def __init__(self, array_path_annot: List[str]):
        self.array_path_to_annot = array_path_annot

    @abstractmethod
    def _get_planes(self, pcd: Pcd, image_colors: str) -> List[Plane]:
        """
        :param pcd: current pcd
        :param image_colors: sequence of colored images
        :return: list of planes in the image
        """
        pass

    def annotate(self, pcd, image_number: int):
        """
        :param pcd: pcd to process
        :param image_number: number of a current pcd in a file list
        :return: pcd with filled field "planes"
        """
        pcd.planes = self._get_planes(pcd, self.array_path_to_annot[image_number])

        return pcd
