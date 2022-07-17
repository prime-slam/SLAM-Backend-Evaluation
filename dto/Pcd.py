import numpy as np


class Pcd:
    def __init__(self, points: np.array):
        self.planes = []
        self.points = points
