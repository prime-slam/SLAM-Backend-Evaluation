import numpy as np
import pytest

from project import config
from tests.data_for_tests.PointCloud import data_paths, ground_truth_data
from project.annotators.AnnotatorPointCloud import AnnotatorPointCloud
from project.pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud


def test_num_annotated_planes():
    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_b = PcdBuilderPointCloud(config.CAMERA_ICL, annot)
    pcd = pcd_b.build_pcd(data_paths.main_data_list[0], 0, verbose=False)

    planes = pcd.planes
    assert len(planes) == len(ground_truth_data.planes_to_test_annotator)


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
    ],
)
def test_planes(file_name, file_num, plane_num):
    reflection = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_b = PcdBuilderPointCloud(config.CAMERA_ICL, annot, reflection)
    pcd = pcd_b.build_pcd(file_name, file_num, True)

    planes = pcd.planes

    plane_to_compare_1 = np.asarray(planes[plane_num].equation)
    plane_to_compare_2 = np.asarray(
        ground_truth_data.planes_to_test_annotator[plane_num]
    )

    np.testing.assert_almost_equal(plane_to_compare_1, plane_to_compare_2)
