import numpy as np

import config
from tests.data_for_tests.PointCloud import data_paths
from PostProcessing import PostProcessing
from annotators.AnnotatorPointCloud import AnnotatorPointCloud
from associators.AssociatorFront import AssociatorFront
from pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud


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

    np.testing.assert_array_equal(max_tracks[:4], [5, 4, 7, 2])
