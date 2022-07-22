import numpy as np
import pytest

import config
from tests.data_for_tests.PointCloud import data_paths, equations
from annotators.AnnotatorPointCloud import AnnotatorPointCloud
from pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud


def test_num_annotated_planes():
    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_b = PcdBuilderPointCloud(config.CAMERA_ICL, annot)
    pcd = pcd_b.build_pcd(data_paths.main_data_list[0], 0)

    planes = pcd.planes
    assert len(planes) == len(equations.planes_to_test_annotator)


@pytest.mark.parametrize("num", [0, 1, 2, 3, 4, 5, 6, 7])
def test_planes(num):
    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_b = PcdBuilderPointCloud(config.CAMERA_ICL, annot)
    pcd = pcd_b.build_pcd(data_paths.main_data_list[0], 0)

    planes = pcd.planes

    plane_to_compare_1 = np.asarray(planes[num].equation)
    plane_to_compare_2 = np.asarray(equations.planes_to_test_annotator[num])

    np.testing.assert_almost_equal(plane_to_compare_1, plane_to_compare_2)
