from abc import ABC, abstractmethod
from typing import List
from project.dto.Pcd import Pcd

import numpy as np


class Associator(ABC):
    """
    Gets correct indices for planes of each image
    """

    @staticmethod
    def make_string_from_array(array: np.array) -> str:
        """
        :param array: array to convert to str
        :return: string with "#" as a separator between the array items
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
