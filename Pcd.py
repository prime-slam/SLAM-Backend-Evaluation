import numpy as np


class Pcd(object):
    def __init__(self, points: np.array):
        self.planes = []
        self.points = points
