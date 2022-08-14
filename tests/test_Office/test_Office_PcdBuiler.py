import numpy as np

from project import config
import open3d as o3d

from project.annotators.AnnotatorImage import AnnotatorImage
from project.pcdBuilders.PcdBuilderOffice import PcdBuilderOffice
from tests.data_for_tests.Office import data_paths


def test_get_points_point_cloud():
    reflection = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_extracted = o3d.io.read_point_cloud(
        "tests/data_for_tests/Office/office00.pcd"
    ).transform(reflection)

    array_to_compare_1 = np.asarray(pcd_extracted.points)

    annot = AnnotatorImage(data_paths.annot_list)
    pcd_builder = PcdBuilderOffice(config.CAMERA_ICL, annot)
    pcd_built = pcd_builder._get_points(data_paths.main_data_list[0])
    array_to_compare_2 = pcd_built.points

    np.testing.assert_almost_equal(array_to_compare_1, array_to_compare_2, decimal=5)
