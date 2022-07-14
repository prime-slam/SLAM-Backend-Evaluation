import math

import numpy as np

import evaluate_ate
import evaluate_rpe


class MeasureError(object):
    def __init__(self,
                 first_node: int,
                 first_gt_node: int,
                 num_of_nodes: int,
                 num_of_all_nodes: int,
                 graph_estimated_state,
                 ds_filename_gt: str,
                 file_name_estimated: str,
                 file_name_gt: str):
        self.first_node = first_node
        self.num_of_nodes = num_of_nodes
        self.num_of_all_nodes = num_of_all_nodes
        self.first_gt_node = first_gt_node
        self.graph_estimated_state = graph_estimated_state
        self.ds_filename_gt = ds_filename_gt
        self.file_name_estimated = file_name_estimated
        self.file_name_gt = file_name_gt

    @staticmethod
    def rotation_matrix_to_quaternion(r_matrix):
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
        translation_str = ' '.join(str(e) for e in translation)
        rotation_str = ' '.join(str(e) for e in rotation)
        return str(timestamp) + ' ' + translation_str + ' ' + rotation_str

    def read_gt_matrices(self):
        in_file = open(self.ds_filename_gt).readlines()  # getting gt data

        all_lines = filter(lambda line: (line != ''), in_file)

        array_with_lines = np.loadtxt(all_lines)
        gt_matrices = np.split(array_with_lines, self.num_of_all_nodes, axis=0)
        return gt_matrices

    def measure_error(self):

        file_to_write_gt = open(self.file_name_gt, 'w')
        file_to_read_gt = open(self.ds_filename_gt)

        bios = 0
        bios_gt = 0

        in_file = file_to_read_gt.readlines()  # mind the first timestamp

        if self.first_node == 0 and self.first_gt_node > 0:
            bios = self.first_gt_node
        elif self.first_gt_node > 0:
            bios_gt = self.first_gt_node

        for line in in_file[self.first_node - bios_gt:self.first_node + self.num_of_nodes - bios_gt - 1]:
            file_to_write_gt.write(line)
        file_to_write_gt.close()

        file_to_write_estimated = open(self.file_name_estimated, 'w')

        estimated_matrices = self.graph_estimated_state[-self.num_of_nodes + bios:]
        for i, matrix in enumerate(estimated_matrices):
            data = self.make_a_string(self.first_node + i + bios, matrix[:3, 3],
                                      self.rotation_matrix_to_quaternion(matrix[:3, :3]))
            file_to_write_estimated.write(data + '\n')
        file_to_write_estimated.close()

        print("ate")
        evaluate_ate.main(self.file_name_gt, self.file_name_estimated)
        print("rpe")
        evaluate_rpe.main(self.file_name_gt, self.file_name_estimated)
