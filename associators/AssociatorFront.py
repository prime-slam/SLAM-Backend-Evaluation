import math
from typing import List
from dto.Pcd import Pcd
from associators.Associator import Associator

import numpy as np


class AssociatorFront(Associator):
    """
    Associates planes with associate_front function with frontend data
    :attribute __set_of_generated_colors: set of used plane colors in order not to allign one color to two different planes
    """

    def __init__(self, set_of_generated_colors=set()):
        self.__set_of_generated_colors = set_of_generated_colors

    def __generate_color(self):
        answ = np.random.randint(255, size=3)
        while self.array_to_string(answ) in self.__set_of_generated_colors:
            answ = self.__generate_color()
        self.__set_of_generated_colors.add(self.array_to_string(answ))

        return answ

    def associate(self, pcd_s: List[Pcd]):
        indx = 0
        for i, prev_pcd in enumerate(pcd_s[:-1]):
            cur_pcd = pcd_s[i + 1]

            if prev_pcd.planes[0].track == -1:
                for i, plane in enumerate(prev_pcd.planes):
                    plane.track = i
                    plane.color = self.__generate_color()
                indx = len(self.__set_of_generated_colors)

            min_bias = 1
            set_prev_used = set()
            map_best_bias = {}

            for cur_plane in cur_pcd.planes:
                for prev_plane in prev_pcd.planes:
                    cur_pair = (prev_plane, cur_plane)  # creating pairs:
                    # (plane from current image,
                    # plane from previous image)
                    deviation = np.dot(
                        prev_plane.equation[:3], cur_plane.equation[:3]
                    )  # cos of an angle: the biggest when planes have the same normal
                    bias = (
                        math.acos(deviation)
                        + math.fabs(prev_plane.equation[-1] - cur_plane.equation[-1])
                        * 10
                    )
                    map_best_bias[cur_pair] = bias
            sorted_map = sorted(map_best_bias.items(), key=lambda item: item[1])
            for pair, bias in sorted_map:
                if bias > min_bias:
                    break
                prev_plane, cur_plane = pair
                diff = len(prev_plane.plane_indices) / len(cur_plane.plane_indices)
                if diff < 0.5 or diff > 2:
                    continue

                if prev_plane.track in set_prev_used or cur_plane.track != -1:
                    continue
                cur_plane.track = prev_plane.track
                cur_plane.color = prev_plane.color
                set_prev_used.add(prev_plane.track)
            for plane in cur_pcd.planes:
                if plane.track == -1:
                    plane.track = indx
                    indx += 1
                    plane.color = self.__generate_color()
        return pcd_s
