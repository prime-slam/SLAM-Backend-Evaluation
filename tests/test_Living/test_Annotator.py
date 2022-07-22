import numpy as np
import pytest

import config
from annotators.AnnotatorImage import AnnotatorImage
from pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from tests.data_for_tests.Living import data_paths, true_data_to_check


def test_num_annotated_planes():
    annot = AnnotatorImage(data_paths.annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcd = pcd_b.build_pcd(data_paths.main_data_list[0], 0)

    planes = pcd.planes
    assert len(planes) == len(true_data_to_check.planes_to_test_annotator)


@pytest.mark.parametrize("num", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
def test_planes(num):
    annot = AnnotatorImage(data_paths.annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcd = pcd_b.build_pcd(data_paths.main_data_list[0], 0)

    planes = pcd.planes

    plane_to_compare_1 = np.asarray(planes[num].equation)
    plane_to_compare_2 = np.asarray(true_data_to_check.planes_to_test_annotator[num])

    np.testing.assert_almost_equal(plane_to_compare_1, plane_to_compare_2)
