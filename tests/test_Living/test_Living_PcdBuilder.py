import numpy as np
import open3d as o3d

from open3d.cpu import pybind
from project import config
from project.annotators.AnnotatorImage import AnnotatorImage
from project.pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from tests.data_for_tests import o3d_camera
from tests.data_for_tests.Living import data_paths


def test_get_points_point_cloud():

    pcd_extracted = o3d.geometry.PointCloud.create_from_depth_image(
        depth=o3d.cpu.pybind.geometry.Image(
            o3d.io.read_image(data_paths.main_data_list[0])
        ),
        intrinsic=o3d_camera.O3D_CAMERA,
        extrinsic=np.eye(4),
        depth_scale=5000,
    )
    reflection = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_extracted.transform(reflection)
    array_to_compare_1 = np.asarray(pcd_extracted.points)

    annot = AnnotatorImage(data_paths.annot_list)
    pcd_builder = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcd_built = pcd_builder._get_points(data_paths.main_data_list[0])
    array_to_compare_2 = pcd_built.points

    np.testing.assert_almost_equal(array_to_compare_1, array_to_compare_2)
