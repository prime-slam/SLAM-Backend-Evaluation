import numpy as np
import pytest

from project import config
from project.annotators.AnnotatorImage import AnnotatorImage
from project.associators.AssociatorAnnot import AssociatorAnnot
from project.pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from tests.data_for_tests.Living import data_paths
from tests.data_for_tests.Living.data_paths import annot_list


plane_colors = [
    [14, 252, 75],
    [56, 34, 229],
    [70, 251, 137],
    [119, 171, 27],
    [132, 29, 75],
    [152, 254, 122],
    [155, 121, 128],
    [156, 244, 150],
    [186, 122, 211],
    [197, 94, 189],
    [214, 164, 69],
    [226, 8, 140],
    [233, 2, 151],
    [237, 154, 228],
    [245, 63, 8],
    [246, 201, 180],
    [68, 69, 34],
]

list_of_params_0 = [[0, plane_colors[i], i + 1] for i in range(17)]
list_of_params_1 = [[1, plane_colors[i], i + 1] for i in range(17)]
list_of_params_2 = [[2, plane_colors[i], i + 1] for i in range(17)]

list_of_params = list_of_params_0 + list_of_params_1 + list_of_params_2


@pytest.mark.parametrize(
    "file_num, color, indx",
    list_of_params,
)
def test_planes(file_num, color, indx):
    pcds = []
    annot = AnnotatorImage(annot_list)
    pcd_b = PcdBuilderLiving(config.CAMERA_ICL, annot)
    pcds.append(pcd_b.build_pcd(data_paths.main_data_list[file_num], file_num))

    associator = AssociatorAnnot()
    associator.associate(pcds)
    for pcd in pcds:
        for plane in pcd.planes:
            if plane.track == indx:
                np.testing.assert_array_equal(plane.color, color)
