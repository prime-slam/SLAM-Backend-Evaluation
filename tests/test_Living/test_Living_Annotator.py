import numpy as np
import pytest

from project import config
from project.annotators.AnnotatorImage import AnnotatorImage
from project.pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
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


list_of_params_0 = [[0, i] for i in range(16)]
list_of_params_1 = [[1, m] for m in range(17)]
list_of_params_2 = [[2, n] for n in range(17)]
list_of_params = list_of_params_0 + list_of_params_1 + list_of_params_2


@pytest.mark.parametrize(
    "file_num, plane_num",
    list_of_params,
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
