import numpy as np

import config
from tests.data_for_tests.Living import data_paths
from PostProcessing import PostProcessing
from annotators.AnnotatorImage import AnnotatorImage
from associators.AssociatorAnnot import AssociatorAnnot
from pcdBuilders.PcdBuilderLiving import PcdBuilderLiving


def test_post_processing():
    annot = AnnotatorImage(data_paths.annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcds = []

    for i, file in enumerate(data_paths.annot_list):
        pcds.append(pcd_b.build_pcd(data_paths.main_data_list[i], i))

    assoc = AssociatorAnnot()
    assoc.associate(pcds)

    post_processing = PostProcessing()
    max_tracks = post_processing.post_process(pcds)

    np.testing.assert_array_equal(
        max_tracks[:4], [8, 13, 11, 3]
    )  # что-то решить с количеством плоскостей
