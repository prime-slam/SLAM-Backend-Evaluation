import numpy as np

from project import config
from tests.data_for_tests.PointCloud import data_paths
from project.PostProcessing import PostProcessing
from project.annotators.AnnotatorPointCloud import AnnotatorPointCloud
from project.associators.AssociatorFront import AssociatorFront
from project.pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud


def test_post_processing():
    reflection = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_b = PcdBuilderPointCloud(config.CAMERA_ICL, annot, reflection)
    pcds = []

    for i, file in enumerate(data_paths.annot_list):
        pcds.append(pcd_b.build_pcd(data_paths.main_data_list[i], i))

    assoc = AssociatorFront()
    assoc.associate(pcds)

    post_processing = PostProcessing()
    max_tracks = post_processing.post_process(pcds)

    np.testing.assert_array_equal(
        max_tracks[: config.MAX_PLANES_COUNT], max_tracks[: config.MAX_PLANES_COUNT]
    )
