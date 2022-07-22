import numpy as np
import pytest

import config
from annotators.AnnotatorImage import AnnotatorImage
from associators.AssociatorAnnot import AssociatorAnnot
from pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from tests.data_for_tests.Living import data_paths
from tests.data_for_tests.Living.data_paths import annot_list


@pytest.mark.parametrize(
    "color, indx",
    [
        [[14, 252, 75], 1],
        [[56, 34, 229], 2],
        [[70, 251, 137], 3],
        [[119, 171, 27], 4],
        [[132, 29, 75], 5],
        [[152, 254, 122], 6],
        [[155, 121, 128], 7],
        [[156, 244, 150], 8],
        [[186, 122, 211], 9],
        [[197, 94, 189], 10],
        [[214, 164, 69], 11],
        [[226, 8, 140], 12],
        [[233, 2, 151], 13],
        [[237, 154, 228], 14],
        [[245, 63, 8], 15],
        [[246, 201, 180], 16],
        [[68, 69, 34], 17],
    ],
)
def test_planes(color, indx):
    pcds = []
    annot = AnnotatorImage(annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)

    for i, image in enumerate(data_paths.main_data_list):
        pcds.append(pcd_b.build_pcd(data_paths.main_data_list[i], i))

    associator = AssociatorAnnot()
    associator.associate(pcds)
    for pcd in pcds:
        for plane in pcd.planes:
            if plane.track == indx:
                np.testing.assert_array_equal(plane.color, color)
