import numpy as np
import pytest

import config
from annotators.AnnotatorImage import AnnotatorImage
from associators.AssociatorAnnot import AssociatorAnnot
from pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from tests.data_for_tests.Living import data_paths
from tests.data_for_tests.Living.data_paths import annot_list


@pytest.mark.parametrize(
    "file_name, file_num, color, indx",
    [
        [data_paths.main_data_list[0], 0, [14, 252, 75], 1],
        [data_paths.main_data_list[0], 0, [56, 34, 229], 2],
        [data_paths.main_data_list[0], 0, [70, 251, 137], 3],
        [data_paths.main_data_list[0], 0, [119, 171, 27], 4],
        [data_paths.main_data_list[0], 0, [132, 29, 75], 5],
        [data_paths.main_data_list[0], 0, [152, 254, 122], 6],
        [data_paths.main_data_list[0], 0, [155, 121, 128], 7],
        [data_paths.main_data_list[0], 0, [156, 244, 150], 8],
        [data_paths.main_data_list[0], 0, [186, 122, 211], 9],
        [data_paths.main_data_list[0], 0, [197, 94, 189], 10],
        [data_paths.main_data_list[0], 0, [214, 164, 69], 11],
        [data_paths.main_data_list[0], 0, [226, 8, 140], 12],
        [data_paths.main_data_list[0], 0, [233, 2, 151], 13],
        [data_paths.main_data_list[0], 0, [237, 154, 228], 14],
        [data_paths.main_data_list[0], 0, [245, 63, 8], 15],
        [data_paths.main_data_list[0], 0, [246, 201, 180], 16],
        [data_paths.main_data_list[0], 0, [68, 69, 34], 17],
        [data_paths.main_data_list[1], 1, [14, 252, 75], 1],
        [data_paths.main_data_list[1], 1, [56, 34, 229], 2],
        [data_paths.main_data_list[1], 1, [70, 251, 137], 3],
        [data_paths.main_data_list[1], 1, [119, 171, 27], 4],
        [data_paths.main_data_list[1], 1, [132, 29, 75], 5],
        [data_paths.main_data_list[1], 1, [152, 254, 122], 6],
        [data_paths.main_data_list[1], 1, [155, 121, 128], 7],
        [data_paths.main_data_list[1], 1, [156, 244, 150], 8],
        [data_paths.main_data_list[1], 1, [186, 122, 211], 9],
        [data_paths.main_data_list[1], 1, [197, 94, 189], 10],
        [data_paths.main_data_list[1], 1, [214, 164, 69], 11],
        [data_paths.main_data_list[1], 1, [226, 8, 140], 12],
        [data_paths.main_data_list[1], 1, [233, 2, 151], 13],
        [data_paths.main_data_list[1], 1, [237, 154, 228], 14],
        [data_paths.main_data_list[1], 1, [245, 63, 8], 15],
        [data_paths.main_data_list[1], 1, [246, 201, 180], 16],
        [data_paths.main_data_list[1], 1, [68, 69, 34], 17],
        [data_paths.main_data_list[2], 2, [14, 252, 75], 1],
        [data_paths.main_data_list[2], 2, [56, 34, 229], 2],
        [data_paths.main_data_list[2], 2, [70, 251, 137], 3],
        [data_paths.main_data_list[2], 2, [119, 171, 27], 4],
        [data_paths.main_data_list[2], 2, [132, 29, 75], 5],
        [data_paths.main_data_list[2], 2, [152, 254, 122], 6],
        [data_paths.main_data_list[2], 2, [155, 121, 128], 7],
        [data_paths.main_data_list[2], 2, [156, 244, 150], 8],
        [data_paths.main_data_list[2], 2, [186, 122, 211], 9],
        [data_paths.main_data_list[2], 2, [197, 94, 189], 10],
        [data_paths.main_data_list[2], 2, [214, 164, 69], 11],
        [data_paths.main_data_list[2], 2, [226, 8, 140], 12],
        [data_paths.main_data_list[2], 2, [233, 2, 151], 13],
        [data_paths.main_data_list[2], 2, [237, 154, 228], 14],
        [data_paths.main_data_list[2], 2, [245, 63, 8], 15],
        [data_paths.main_data_list[2], 2, [246, 201, 180], 16],
        [data_paths.main_data_list[2], 2, [68, 69, 34], 17],
    ],
)
def test_planes(file_name, file_num, color, indx):
    pcds = []
    annot = AnnotatorImage(annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcds.append(pcd_b.build_pcd(file_name, file_num))

    associator = AssociatorAnnot()
    associator.associate(pcds)
    for pcd in pcds:
        for plane in pcd.planes:
            if plane.track == indx:
                np.testing.assert_array_equal(plane.color, color)
