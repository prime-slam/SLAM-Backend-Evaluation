from abc import ABC, abstractmethod
from typing import List
from dto.Pcd import Pcd


class Annotator(ABC):
    def __init__(self, array_path_annot: List[str]):
        self.array_path_to_annot = array_path_annot

    @abstractmethod
    def _get_planes(self, pcd: Pcd, image_colors: str):
        pass

    def annotate(self, pcd, image_number: int):
        pcd.planes = self._get_planes(pcd, self.array_path_to_annot[image_number])

        return pcd
