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


@pytest.mark.parametrize(
    "file_name, file_num, plane_num",
    [
        [data_paths.main_data_list[0], 0, 0],
        [data_paths.main_data_list[0], 0, 1],
        [data_paths.main_data_list[0], 0, 2],
        [data_paths.main_data_list[0], 0, 3],
        [data_paths.main_data_list[0], 0, 4],
        [data_paths.main_data_list[0], 0, 5],
        [data_paths.main_data_list[0], 0, 6],
        [data_paths.main_data_list[0], 0, 7],
        [data_paths.main_data_list[0], 0, 8],
        [data_paths.main_data_list[0], 0, 9],
        [data_paths.main_data_list[0], 0, 10],
        [data_paths.main_data_list[0], 0, 11],
        [data_paths.main_data_list[0], 0, 12],
        [data_paths.main_data_list[0], 0, 13],
        [data_paths.main_data_list[0], 0, 14],
        [data_paths.main_data_list[0], 0, 15],
    ],
)
def test_planes(file_name, file_num, plane_num):
    annot = AnnotatorImage(data_paths.annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcd = pcd_b.build_pcd(file_name, file_num)

    planes = pcd.planes

    plane_to_compare_1 = np.asarray(planes[plane_num].equation)
    plane_to_compare_2 = np.asarray(
        true_data_to_check.planes_to_test_annotator[plane_num]
    )
    np.testing.assert_almost_equal(plane_to_compare_1, plane_to_compare_2)
