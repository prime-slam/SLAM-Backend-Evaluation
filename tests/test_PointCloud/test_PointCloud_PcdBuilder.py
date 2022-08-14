import numpy as np
import open3d as o3d
import pytest

from project import config
from tests.data_for_tests.PointCloud import data_paths
from project.annotators.AnnotatorPointCloud import AnnotatorPointCloud
from project.pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud


@pytest.mark.parametrize("file_num", [0, 1, 2])
def test_get_points_point_cloud(file_num):
    reflection = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    pcd_extracted = o3d.io.read_point_cloud(
        data_paths.main_data_list[file_num]
    ).transform(reflection)
    array_to_compare_1 = np.asarray(pcd_extracted.points) / 1000

    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_builder = PcdBuilderPointCloud(config.CAMERA_ICL, annot)
    pcd_built = pcd_builder._get_points(data_paths.main_data_list[file_num])
    array_to_compare_2 = pcd_built.points

    np.testing.assert_array_equal(array_to_compare_1, array_to_compare_2)
