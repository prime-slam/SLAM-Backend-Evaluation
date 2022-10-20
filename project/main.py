import argparse
import csv
import os
from typing import List, Tuple

import numpy as np

from project import read_office, config
from project.SLAMGraph import SLAMGraph
from project.Visualisation import Visualisation
from project.annotators.AnnotatorImage import AnnotatorImage
from project.annotators.AnnotatorPointCloud import AnnotatorPointCloud

from project.associators.AssociatorAnnotImg import AssociatorAnnotImg
from project.associators.AssociatorAnnotNpy import AssociatorAnnotNpy
from project.associators.AssociatorFront import AssociatorFront
from project.measurements.MeasureError import MeasureError
from project.pcdBuilders.PcdBuilderKitti import PcdBuilderKitti
from project.pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from project.pcdBuilders.PcdBuilderOffice import PcdBuilderOffice
from project.pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud
from project.postprocessing.EnoughPlanesDetector import EnoughPlanesDetector
from project.postprocessing.PlaneInfoPrinter import PlaneInfoPrinter
from project.postprocessing.PlaneRemover import PlaneRemover
from project.postprocessing.SmallPlanesFilter import SmallPlanesFilter
from project.utils.intervals import load_evaluation_intervals, dump_evaluation_intervals, ids_list_to_intervals

FORMAT_ICL_TUM = 1
FORMAT_ICL = 2
FORMAT_PCD = 3
FORMAT_KITTI = 4
FORMAT_CARLA = 5


class TumFilenameComparator(str):
    def __lt__(self, other):
        items = self[:-4].split(".")
        other_items = other[:-4].split(".")
        for i, item in enumerate(items):
            if int(item) != int(other_items[i]):
                return int(item) < int(other_items[i])

        return True


def create_data_list_kitti(main_data_path: str):
    depth = os.listdir(main_data_path)
    depth = sorted(depth, key=lambda x: int(x[-10:-4]))
    main_data_list = list(map(lambda x: os.path.join(main_data_path, x), depth))
    return main_data_list


def create_data_list_living(main_data_path: str):
    depth = os.listdir(main_data_path)
    depth = sorted(depth, key=TumFilenameComparator)
    main_data_list = list(map(lambda x: os.path.join(main_data_path, x), depth))
    return main_data_list


def create_annot_list_office(main_data_path: str):
    depth = os.listdir(main_data_path)
    depth = sorted(depth, key=lambda x: int(x[9:-4]))
    main_data_list = list(map(lambda x: os.path.join(main_data_path, x), depth))
    return main_data_list


def create_main_lists_office(main_data_path: str):
    png_files, depths_files = read_office.provide_filenames(main_data_path)
    return png_files, depths_files


def create_main_and_annot_list_carla(main_data_path: str):
    files = os.listdir(main_data_path)
    annot_list = list(map(lambda x: os.path.join(main_data_path, x), filter(lambda x: x.endswith(".npy"), files)))
    main_data_list = list(map(lambda x: os.path.join(main_data_path, x), filter(lambda x: x.endswith(".pcd"), files)))

    return annot_list, main_data_list


def create_main_and_annot_list(main_data_path: str):
    annot_list = []
    main_data_list = []

    folders = os.listdir(main_data_path)

    for j, folder in enumerate(folders):
        cur_path = os.path.join(main_data_path, folder)
        npy, pcd = os.listdir(cur_path)

        cur_path_to_annot = os.path.join(cur_path, npy)
        cur_path_to_main_data = os.path.join(cur_path, pcd)

        annot_list.append(cur_path_to_annot)
        main_data_list.append(cur_path_to_main_data)
    return annot_list, main_data_list


def main(
    main_data_path: str,
    annot_path: str,
    input_format_code: int,
    intervals_source_path: str,
    first_gt_node: int,
    ds_filename_gt: str,
):
    camera = config.CAMERA_ICL

    if input_format_code == FORMAT_ICL_TUM or input_format_code == FORMAT_ICL:
        if input_format_code == FORMAT_ICL_TUM:
            main_data_list = create_data_list_living(main_data_path)
            annot_list = create_data_list_living(annot_path)
        else:
            annot_list = create_annot_list_office(annot_path)
            png_list, main_data_list = create_main_lists_office(main_data_path)

        if input_format_code == FORMAT_ICL_TUM:
            annot = AnnotatorImage(annot_list, main_data_list)
            pcd_b = PcdBuilderLiving(camera, annot)
        else:
            annot = AnnotatorImage(annot_list, main_data_list, is_office=True)
            pcd_b = PcdBuilderOffice(camera, annot)
        associator = AssociatorAnnotImg()
    elif input_format_code == FORMAT_PCD:
        annot_list, main_data_list = create_main_and_annot_list(main_data_path)
        annot = AnnotatorPointCloud(annot_list)
        reflection = np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        pcd_b = PcdBuilderPointCloud(
            camera, annot, reflection
        )  # reflection is needed due to dataset (icl nuim) particularities
        associator = AssociatorFront()
    elif input_format_code == FORMAT_CARLA:
        annot_list, main_data_list = create_main_and_annot_list_carla(main_data_path)
        annot = AnnotatorPointCloud(annot_list)
        reflection = np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        pcd_b = PcdBuilderPointCloud(
            camera, annot, reflection, scale=1
        )  # reflection is needed due to dataset (icl nuim) particularities
        associator = AssociatorAnnotNpy()
    else:
        main_data_list = create_data_list_kitti(main_data_path)
        annot_list = create_data_list_kitti(annot_path)
        annot = AnnotatorPointCloud(annot_list)
        pcd_b = PcdBuilderKitti(camera, annot, None)
        associator = AssociatorAnnotNpy()

    total_frames = len(main_data_list)

    # TODO: 0,323 living
    # TODO: 1203,1285 living
    # TODO: 1300,1508 living


    # in not normalized format (0-255 for RGB)
    ignore_color_strs = [
        # # Associator.make_string_from_array([156, 244, 150]),  # подушка-сидушка левая  --- не влияет на первом интервале (380 -- 500)
        # # Associator.make_string_from_array([152, 254, 122]),  # подушка-сидушка правая  --- не влияет на первом интервале (380 -- 500)
        # # Associator.make_string_from_array([197,  94, 189]),  # левый подлокотник --- не влияет на первом интервале (380 -- 500)
        # Associator.make_string_from_array([14, 252, 75]), # подушка-спинка левая --- там разрывы по глубине, потолок сьезжает ниже (380 -- 500)
        # # Associator.make_string_from_array([119, 171, 27]),  # подушка-спинка правая --- не влияет на первом интервале (380 -- 500)
        # Associator.make_string_from_array([245, 63, 8]),  # пол -- едет потолок по вертикали (380 -- 500)
        # Associator.make_string_from_array([233, 2, 151]), # подушка-нижняя нижняя --- вообще стемная, цепляет немного загиб, поэтому едет потолок по вертикали (380 -- 500)
        # Associator.make_string_from_array([226, 8, 140]),  # подушка-нижняя левая --- едет потолок по вертикали (380 -- 500)
        # Associator.make_string_from_array([237, 154, 228]),  # подушка-нижняя правая --- едет потолок по вертикали (380 -- 500)
        # # Associator.make_string_from_array([57, 181, 174]),  # тумба левая внутренность --- не влияет на первом интервале (380 -- 500)
        # Associator.make_string_from_array([155, 121, 128]), # подлокотник вертикальная плоскость -- ломало левую стену по горизонту (380 -- 500)
    ]

    pcd_enough_planes_ids = []
    if intervals_source_path is None or not os.path.exists(intervals_source_path):
        for i, file in enumerate(main_data_list):
            pcd = pcd_b.build_pcd(file, i)

            pcd = SmallPlanesFilter.filter(pcd)
            pcd = PlaneRemover.remove_by_colors(pcd, ignore_color_strs)

            print("Frame: {}".format(i))
            is_enough = EnoughPlanesDetector.has_enough_planes(pcd)
            if is_enough:
                pcd_enough_planes_ids.append(i)

        enough_intervals = ids_list_to_intervals(pcd_enough_planes_ids)
        dump_evaluation_intervals("intervals.csv", enough_intervals)
    else:
        enough_intervals = load_evaluation_intervals(intervals_source_path, total_frames)

    for interval in enough_intervals:
        pcds = []
        for frame_id in range(interval[0], interval[1] + 1):
            pcd = pcd_b.build_pcd(main_data_list[frame_id], frame_id)

            pcd = SmallPlanesFilter.filter(pcd)
            pcd = PlaneRemover.remove_by_colors(pcd, ignore_color_strs)

            print("Frame: {}".format(frame_id))
            pcds.append(pcd)

        associator.associate(pcds)

        PlaneInfoPrinter.print_planes_info(pcds)

        # max_tracks = PostProcessing.post_process(pcds)

        slam_graph = SLAMGraph()
        # graph_estimated_state = slam_graph.estimate_graph(pcds, max_tracks)
        graph_estimated_state = slam_graph.estimate_graph(pcds)

        # measure_error = MeasureError(ds_filename_gt, len(annot_list), num_of_nodes)
        # measure_error.measure_error(first_node, first_gt_node, graph_estimated_state)
        #
        visualisation = Visualisation(graph_estimated_state)
        visualisation.visualize(pcds, graph_estimated_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarks a trajectory, built by an algorithm"
    )
    parser.add_argument(
        "main_data", type=str, help="Directory where main information files are stored"
    )
    parser.add_argument(
        "annot", type=str, help="Directory where color images are stored"
    )
    parser.add_argument(
        "--format",
        type=int,
        choices=[FORMAT_ICL_TUM, FORMAT_ICL, FORMAT_PCD, FORMAT_KITTI, FORMAT_CARLA],
        help="living room = 1, office = 2, point clouds = 3, kitti = 4, carla = 5",
    )
    parser.add_argument(
        "--evaluate_intervals_source_path",
        type=str,
        default=None,
        help="Path to csv with intervals of frames to evaluate on"
    )
    parser.add_argument(
        "first_gt_node", type=int, help="From what node gt references start"
    )
    parser.add_argument(
        "ds_filename_gt", type=str, help="Filename of a file with gt references"
    )
    args = parser.parse_args()

    main(
        args.main_data,
        args.annot,
        args.format,
        args.evaluate_intervals_source_path,
        args.first_gt_node,
        args.ds_filename_gt,
    )
