import numpy as np
import pytest

import config
from annotators.AnnotatorImage import AnnotatorImage
from pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from tests.data_for_tests.Living import data_paths, ground_truth_data


@pytest.mark.parametrize(
    "file_num",
    [
        0,
        1,
        2,
    ],
)
def test_num_annotated_planes(file_num):
    annot = AnnotatorImage(data_paths.annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcd = pcd_b.build_pcd(data_paths.main_data_list[file_num], file_num)

    planes = pcd.planes
    assert len(planes) == len(ground_truth_data.planes_to_test_annotator[file_num])


@pytest.mark.parametrize(
    "file_num, plane_num",
    [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [0, 14],
        [0, 15],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [1, 6],
        [1, 7],
        [1, 8],
        [1, 9],
        [1, 10],
        [1, 11],
        [1, 12],
        [1, 13],
        [1, 14],
        [1, 15],
        [1, 16],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],
        [2, 15],
        [2, 16],
    ],
)
def test_planes(file_num, plane_num):
    annot = AnnotatorImage(data_paths.annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcd = pcd_b.build_pcd(data_paths.main_data_list[file_num], file_num)

    planes = pcd.planes

    plane_to_compare_1 = np.asarray(planes[plane_num].equation)
    plane_to_compare_2 = np.asarray(
        ground_truth_data.planes_to_test_annotator[file_num][plane_num]
    )
    np.testing.assert_almost_equal(plane_to_compare_1, plane_to_compare_2)
