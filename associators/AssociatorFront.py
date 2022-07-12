import math

import numpy as np

from associators.Associator import Associator


class AssociatorFront(Associator):
    def __init__(self, pcd_s):
        super().__init__(pcd_s)

    def __generate_color(self, set_of_generated_colors):
        answ = np.random.randint(255, size=3)
        while self.array_to_string(answ) in set_of_generated_colors:
            answ = self.__generate_color(set_of_generated_colors)
        set_of_generated_colors.add(self.array_to_string(answ))

        return answ

    def associate(self):
        set_of_generated_colors = set()
        for i,_ in enumerate(self.__pcd_s):
            prev_pcd = self.__pcd_s[i]
            cur_pcd = self.__pcd_s[i+1]

            if prev_pcd.planes[0].track == -1:
                for plane, i in enumerate(cur_pcd.planes()):
                    plane.track = i
            else:
                indx = len(prev_pcd.planes())
                for plane in cur_pcd.planes():
                    min_bias = 10
                    most_correct_plane = -1
                    for prev_plane in prev_pcd.planes:
                        deviation = np.dot(prev_plane.equation[:3],
                                           plane.equation[:3])  # косинус угла: наибольший, когда плоскости совпадают
                        bias = math.acos(deviation) + math.fabs(prev_plane.equation[-1] - plane.equation[-1])
                        if bias < min_bias:
                            min_bias = bias
                            most_correct_plane = prev_plane
                    if most_correct_plane == -1:
                        plane.track = indx
                        plane.color = self.__generate_color(set_of_generated_colors)
                    elif most_correct_plane != -1:
                        plane.track = most_correct_plane.track
                        plane.color = most_correct_plane.color

        return self.__pcd_s
