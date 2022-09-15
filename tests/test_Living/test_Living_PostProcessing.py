import numpy as np

from project import config
from tests.data_for_tests.Living import data_paths
from project.PostProcessing import PostProcessing
from project.annotators.AnnotatorImage import AnnotatorImage
from project.associators.AssociatorAnnot import AssociatorAnnot
from project.pcdBuilders.PcdBuilderLiving import PcdBuilderLiving


def test_post_processing():
    annot = AnnotatorImage(data_paths.annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcds = []

    for i, file in enumerate(data_paths.annot_list):
        pcds.append(pcd_b.build_pcd(data_paths.main_data_list[i], i, verbose=False))

    assoc = AssociatorAnnot()
    assoc.associate(pcds)

    post_processing = PostProcessing()
    max_tracks = post_processing.post_process(pcds, verbose=False)

    np.testing.assert_array_equal(
        max_tracks[: config.MAX_PLANES_COUNT], max_tracks[: config.MAX_PLANES_COUNT]
    )
