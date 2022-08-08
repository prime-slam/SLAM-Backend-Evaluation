import numpy as np

from scripts import config
from tests.data_for_tests.PointCloud import data_paths
from scripts.PostProcessing import PostProcessing
from scripts.annotators.AnnotatorPointCloud import AnnotatorPointCloud
from scripts.associators.AssociatorFront import AssociatorFront
from scripts.pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud


def test_post_processing():
    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_b = PcdBuilderPointCloud(config.CAMERA_ICL, annot)
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
