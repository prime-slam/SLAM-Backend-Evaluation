import pytest
import numpy as np

from tests.data_for_tests.PointCloud import data_paths
from scripts import config
from scripts.annotators.AnnotatorPointCloud import AnnotatorPointCloud
from scripts.associators.AssociatorFront import AssociatorFront
from scripts.pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud
from tests.data_for_tests.PointCloud.ground_truth_data import associated_planes


@pytest.mark.parametrize(
    "file_num, indx",
    [
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 8],
        [1, 4],
        [1, 5],
        [1, 6],
        [1, 7],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 9],
        [2, 8],
        [2, 4],
        [2, 6],
    ],
)
def test_associated_planes(file_num, indx):
    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_b = PcdBuilderPointCloud(config.CAMERA_ICL, annot)
    pcd_s = []

    for i, image in enumerate(data_paths.main_data_list):
        pcd_s.append(pcd_b.build_pcd(image, i))

    associator = AssociatorFront()
    pcd_s = associator.associate(pcd_s)
    for plane in pcd_s[file_num].planes:
        if plane.track == indx:
            np.testing.assert_almost_equal(
                plane.equation, associated_planes[file_num - 1][indx]
            )
