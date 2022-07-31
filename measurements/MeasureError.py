import math
import os

import numpy as np

from measurements import evaluate_ate, evaluate_rpe


class MeasureError:
    """
    evaluates ate and rpe errors with TUM scripts for evaluating
    :attribute ds_filename_gt: directory with ground truth data
    :attribute num_of_all_nodes: number of all the nodes
    :attribute __file_to_write_estimated: file name where to all the needed estimated data will be writen
    :attribute __file_to_write_gt: file name where all the needed ground truth data will be writen
    """

    def __init__(self, ds_filename_gt: str, num_of_all_nodes: int):
        self.ds_filename_gt = ds_filename_gt
        self.num_of_all_nodes = num_of_all_nodes
        self.__file_to_write_estimated = "measure_error_estimated.txt"
        self.__file_to_write_gt = "measure_error_gt.txt"

    @staticmethod
    def rotation_matrix_to_quaternion(r_matrix):
        """
        :param r_matrix: rotation as a matrix
        :return: rotation as a quaternion
        """
        # First row of the rotation matrix
        r00 = r_matrix[0, 0]
        r01 = r_matrix[0, 1]
        r02 = r_matrix[0, 2]

        # Second row of the rotation matrix
        r10 = r_matrix[1, 0]
        r11 = r_matrix[1, 1]
        r12 = r_matrix[1, 2]

        # Third row of the rotation matrix
        r20 = r_matrix[2, 0]
        r21 = r_matrix[2, 1]
        r22 = r_matrix[2, 2]

        tr = r00 + r11 + r22

        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * s
            qx = (r21 - r12) / s
            qy = (r02 - r20) / s
            qz = (r10 - r01) / s
        elif r00 > r11 and r00 > r22:
            s = math.sqrt(1.0 + r00 - r11 - r22) * 2
            qw = (r21 - r12) / s
            qx = 0.25 * s
            qy = (r01 + r10) / s
            qz = (r02 + r20) / s

        elif r11 > r22:
            s = math.sqrt(1.0 + r11 - r00 - r22) * 2
            qw = (r02 - r20) / s
            qx = (r01 + r10) / s
            qy = 0.25 * s
            qz = (r12 + r21) / s
        else:
            s = math.sqrt(1.0 + r22 - r00 - r11) * 2
            qw = (r10 - r01) / s
            qx = (r02 + r20) / s
            qy = (r12 + r21) / s
            qz = 0.25 * s

        q = [qx, qy, qz, qw]

        return q

    @staticmethod
    def make_a_string(timestamp, translation, rotation):
        """
        :param timestamp: timestamp
        :param translation: translation as a vector
        :param rotation: rotation as a quaternion
        :return: string of a concatenated params
        """
        translation_str = " ".join(str(e) for e in translation)
        rotation_str = " ".join(str(e) for e in rotation)
        return str(timestamp) + " " + translation_str + " " + rotation_str

    def read_gt_matrices(self):
        """
        :return: array of splited matrices
        """
        in_file = open(self.ds_filename_gt).readlines()  # getting gt data

        all_lines = filter(lambda line: (line != ""), in_file)

        array_with_lines = np.loadtxt(all_lines)
        gt_matrices = np.split(array_with_lines, self.num_of_all_nodes, axis=0)
        return gt_matrices

    def measure_error(self, first_node: int, first_gt_node: int, graph_estimated_state):
        """
        :param first_node: first node of data sequence
        :param first_gt_node: first node of ground truth sequence
        :param graph_estimated_state: estimated graph
        :return: writes measured errors
        """
        num_of_nodes = self.num_of_all_nodes

        file_to_write_gt = open(self.__file_to_write_gt, "w")
        file_to_read_gt = open(self.ds_filename_gt)

        bios = 0
        bios_gt = 0

        in_file = file_to_read_gt.readlines()  # mind the first timestamp

        if first_node == 0 and first_gt_node > 0:
            bios = first_gt_node
        elif first_gt_node > 0:
            bios_gt = first_gt_node

        for line in in_file[
            first_node - bios_gt : first_node + num_of_nodes - bios_gt - 1
        ]:
            file_to_write_gt.write(line)
        file_to_write_gt.close()

        file_to_write_estimated = open(self.__file_to_write_estimated, "w")

        estimated_matrices = graph_estimated_state[-num_of_nodes + bios :]
        for i, matrix in enumerate(estimated_matrices):
            data = self.make_a_string(
                first_node + i + bios,
                matrix[:3, 3],
                self.rotation_matrix_to_quaternion(matrix[:3, :3]),
            )
            file_to_write_estimated.write(data + "\n")
        file_to_write_estimated.close()

        print("ate")
        evaluate_ate.main(self.__file_to_write_gt, self.__file_to_write_estimated)
        print("rpe")
        evaluate_rpe.main(self.__file_to_write_gt, self.__file_to_write_estimated)

        os.remove(self.__file_to_write_gt)
        os.remove(self.__file_to_write_estimated)
