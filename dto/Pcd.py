import numpy as np


class Pcd:
    """
    A class to represent a point cloud
    :attribute planes: planes of an image
    :attribute points: all points of an image
    """

    def __init__(self, points: np.array):
        self.planes = []
        self.points = points
