import numpy as np


class Plane(object):
    def __init__(self, equation, track: int, color, indices):
        self.equation = equation
        self.track = track
        self.color = color
        self.plane_indices = indices
    @staticmethod
    def get_normal(points):
        c = np.mean(points, axis=0)
        A = np.array(points) - c
        eigvals, eigvects = np.linalg.eig(A.T @ A)
        min_index = np.argmin(eigvals)
        n = eigvects[:, min_index]

        d = -np.dot(n, c)
        normal = int(np.sign(d)) * n
        d *= np.sign(d)
        return np.asarray([normal[0], normal[1], normal[2], d])

