import numpy as np

from Pcd import Pcd


class PostProcessing(object):
    def __init__(self, pcds: list[Pcd]):
        self.array_indx = np.zeros(4)
        self.array_vals = np.zeros(4)
        self.pcd_s = pcds

    def __get_best_planes(self):
        for pcd in self.pcd_s:
            for plane in pcd.planes:
                num_of_points = len(plane.indices)
                if num_of_points > self.array_vals[0]:
                    self.array_vals[3] = self.array_vals[2]
                    self.array_indx[3] = self.array_indx[2]

                    self.array_vals[2] = self.array_vals[1]
                    self.array_indx[2] = self.array_indx[1]

                    self.array_vals[1] = self.array_vals[0]
                    self.array_indx[1] = self.array_indx[0]

                    self.array_vals[0] = num_of_points
                    self.array_indx[0] = plane.track

                elif num_of_points > self.array_vals[1]:
                    self.array_vals[3] = self.array_vals[2]
                    self.array_indx[3] = self.array_indx[2]

                    self.array_vals[2] = self.array_vals[1]
                    self.array_indx[2] = self.array_indx[1]

                    self.array_vals[1] = num_of_points
                    self.array_indx[1] = plane.track

                elif num_of_points > self.array_vals[2]:
                    self.array_vals[3] = self.array_vals[2]
                    self.array_indx[3] = self.array_indx[2]

                    self.array_vals[2] = num_of_points
                    self.array_indx[2] = plane.track

                elif num_of_points > self.array_vals[3]:
                    self.array_vals[3] = num_of_points
                    self.array_indx[3] = plane.track
        return self.array_vals, self.array_indx

    def post_process(self):
        _, arr_indx = self.__get_best_planes()
        for pcd in self.pcd_s:
            for plane in pcd.planes:
                if plane.track not in arr_indx:  # can we do like that?
                    pcd.planes.remove(plane)
        return self.pcd_s
