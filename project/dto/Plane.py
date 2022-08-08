import numpy as np


class Plane:
    """
    A class to represent a plane
    :attribute equation: equation of a plane
    :attribute track: track of a plane
    :attribute color: color to use on planes of this track
    :attribute plane_indices: indices of points that belong to a plane
    """

    def __init__(self, equation, track: int, color, indices):
        self.equation = equation
        self.track = track
        self.color = color
        self.plane_indices = indices

    @staticmethod
    def get_equation(points):
        """
        :param points: all points of a plane
        :return: equation of a plane
        """
        c = np.mean(points, axis=0)
        A = np.array(points) - c
        eigvals, eigvects = np.linalg.eig(A.T @ A)
        min_index = np.argmin(eigvals)
        n = eigvects[:, min_index]

        d = -np.dot(n, c)
        normal = int(np.sign(d)) * n
        d *= np.sign(d)
        return np.asarray([normal[0], normal[1], normal[2], d])
