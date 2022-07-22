import numpy as np
import open3d as o3d

# import config
# import tests_data_paths
# from annotators.AnnotatorPointCloud import AnnotatorPointCloud
# from main import create_main_and_annot_list
# from pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud
import config
from tests.data_for_tests.PointCloud import data_paths
from annotators.AnnotatorPointCloud import AnnotatorPointCloud
from pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud


def test_get_points_point_cloud():

    pcd_extracted = o3d.io.read_point_cloud(data_paths.main_data_list[0])
    array_to_compare_1 = np.asarray(pcd_extracted.points) / 1000

    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_builder = PcdBuilderPointCloud(config.CAMERA_ICL, annot)
    pcd_built = pcd_builder._get_points(data_paths.main_data_list[0])
    array_to_compare_2 = pcd_built.points

    np.testing.assert_array_equal(array_to_compare_1, array_to_compare_2)
