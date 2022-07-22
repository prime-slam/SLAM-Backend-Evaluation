import config
import pytest
from tests.data_for_tests.PointCloud import data_paths
import numpy as np
from annotators.AnnotatorPointCloud import AnnotatorPointCloud
from associators.AssociatorFront import AssociatorFront
from pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud


@pytest.mark.parametrize(
    "a, expected_result",
    [
        (
            0,
            [
                0.01982171347182044,
                -0.002390449634066292,
                -0.9998006728471371,
                3.373238599022399,
            ],
        ),
        (
            1,
            [
                0.9997419043986149,
                0.0009914237793657245,
                0.022696732547231774,
                1.0542992005779215,
            ],
        ),
        (
            2,
            [
                0.000870519364763496,
                -0.9999893846600166,
                0.004524683780847047,
                1.108547915195423,
            ],
        ),
        (
            4,
            [
                -0.0003367920120127402,
                0.9999295441754285,
                -0.01186563341168601,
                0.8960512809579099,
            ],
        ),
        (
            5,
            [
                0.03558406010991272,
                0.9992695347740388,
                0.013934544792360469,
                1.3440632539036783,
            ],
        ),
        (
            6,
            [
                0.013552659475391292,
                0.19686218458629492,
                -0.9803374958151176,
                3.12614055054387,
            ],
        ),
        (
            7,
            [
                0.023112305093756474,
                -0.0011328583271076446,
                -0.999732233143087,
                2.310440825876544,
            ],
        ),
    ],
)
def test_asserted_planes(a, expected_result):
    annot = AnnotatorPointCloud(data_paths.annot_list)
    pcd_b_1 = PcdBuilderPointCloud(config.CAMERA_ICL, annot)
    pcd_1 = pcd_b_1.build_pcd(data_paths.main_data_list[0], 0)

    pcd_b_2 = PcdBuilderPointCloud(config.CAMERA_ICL, annot)
    pcd_2 = pcd_b_2.build_pcd(data_paths.main_data_list[1], 1)

    assoc = AssociatorFront()
    pcd_s = assoc.associate([pcd_1, pcd_2])
    np.testing.assert_almost_equal(pcd_s[1].planes[a].equation, expected_result)
