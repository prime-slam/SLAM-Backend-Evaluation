from abc import ABC, abstractmethod
from typing import List

import numpy as np

from dto.Pcd import Pcd


class Associator(ABC):
    @staticmethod
    def array_to_string(array: np.array) -> str:
        """
        :param array: array of ints
        :return: string with "#" as a separator between the numbers
        """
        channels = [str(num) for num in array]
        string = "#".join(channels)

        return string

    @abstractmethod
    def associate(self, pcd_s: List[Pcd]) -> List[Pcd]:
        """
        :param pcd_s: List of pcds
        :return: LIst of pcds with associated planes
        """
        pass
