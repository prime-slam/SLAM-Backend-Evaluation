import numpy as np
from numpy import array
from open3d.cpu import pybind

import config
import open3d as o3d
from annotators.AnnotatorImage import AnnotatorImage
from pcdBuilders.PcdBuilderOffice import PcdBuilderOffice
from tests.data_for_tests.Office import data_paths


def test_get_points_point_cloud():
    pcd_extracted = o3d.io.read_point_cloud(
        "C:\\work\\GitHub\\PointCloudsAndStuff\\tests\\data_for_tests\\Office\\office00.pcd"
    )

    array_to_compare_1 = np.asarray(pcd_extracted.points)

    annot = AnnotatorImage(data_paths.annot_list)
    pcd_builder = PcdBuilderOffice(config.CAMERA_ICL, annot)
    pcd_built = pcd_builder._get_points(data_paths.main_data_list[0])
    array_to_compare_2 = pcd_built.points

    np.testing.assert_almost_equal(array_to_compare_1, array_to_compare_2, decimal=5)
