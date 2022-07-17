import math
from typing import List
from dto.Pcd import Pcd
from associators.Associator import Associator

import numpy as np


class AssociatorFront(Associator):
    def __init__(self, set_of_generated_colors=set()):
        super().__init__()
        self.__set_of_generated_colors = set_of_generated_colors

    def __generate_color(self):
        answ = np.random.randint(255, size=3)
        while self.array_to_string(answ) in self.__set_of_generated_colors:
            answ = self.__generate_color()
        self.__set_of_generated_colors.add(self.array_to_string(answ))

        return answ

    def associate(self, pcd_s: List[Pcd]):
        set_of_generated_colors = set()
        indx = 0
        for i, prev_pcd in enumerate(pcd_s[:-1]):
            cur_pcd = pcd_s[i + 1]

            if prev_pcd.planes[0].track == -1:
                for i, plane in enumerate(prev_pcd.planes):
                    plane.track = i
                    plane.color = self.__generate_color()
                indx = len(set_of_generated_colors)

            min_bias = 1

            for cur_plane in cur_pcd.planes:
                map_best_bias = {}
                for prev_plane in prev_pcd.planes:
                    cur_pair = (prev_plane, cur_plane)
                    deviation = np.dot(prev_plane.equation[:3],
                                       cur_plane.equation[:3])  # косинус угла: наибольший, когда плоскости совпадают
                    bias = math.acos(deviation) + math.fabs(prev_plane.equation[-1] - cur_plane.equation[-1])
                    map_best_bias[cur_pair] = bias
                sorted_map = sorted(map_best_bias.items(), key=lambda item: item[1])
                if sorted_map[0][1] < min_bias and sorted_map[0][0][1].track == -1:
                    cur_plane.track = sorted_map[0][0][0].track
                    cur_plane.color = sorted_map[0][0][0].color
                elif sorted_map[0][1] > min_bias:
                    cur_plane.track = indx
                    indx += 1
                    cur_plane.color = self.__generate_color()

        #
        #

            print("association " + str(i))

        return pcd_s
