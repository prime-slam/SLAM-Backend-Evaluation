import argparse
import os

import numpy as np

from project import read_office, config
from project.SLAMGraph import SLAMGraph
from project.annotators.AnnotatorImage import AnnotatorImage
from project.annotators.AnnotatorPointCloud import AnnotatorPointCloud
from project.associators.AssociatorAnnot import AssociatorAnnot
from project.associators.AssociatorFront import AssociatorFront
from project.measurements.MeasureError import MeasureError
from project.pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from project.pcdBuilders.PcdBuilderOffice import PcdBuilderOffice
from project.pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointCloud
from project.PostProcessing import PostProcessing
from project.Visualisation import Visualisation


def create_data_list_living(main_data_path: str):
    depth = os.listdir(main_data_path)
    depth = sorted(depth, key=lambda x: int(x[:-4]))
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
    which_format: int,
    first_node: int,
    first_gt_node: int,
    num_of_nodes: int,
    ds_filename_gt: str,
):

    camera = config.CAMERA_ICL
    pcds = []

    if which_format == 1 or which_format == 2:
        if which_format == 1:
            main_data_list = create_data_list_living(main_data_path)[
                first_node : first_node + num_of_nodes
            ]
            annot_list = create_data_list_living(annot_path)[
                first_node : first_node + num_of_nodes
            ]
        else:
            annot_list = create_annot_list_office(annot_path)
            png_list, main_data_list = create_main_lists_office(main_data_path)
            png_list = png_list[first_node : first_node + num_of_nodes]
            main_data_list = main_data_list[first_node : first_node + num_of_nodes]
            print(main_data_list)
        annot = AnnotatorImage(annot_list)
        if which_format == 1:
            pcd_b = PcdBuilderLiving(camera, annot)
        else:
            pcd_b = PcdBuilderOffice(camera, annot)

        for i, image in enumerate(main_data_list):
            pcds.append(pcd_b.build_pcd(main_data_list[i], i))

        associator = AssociatorAnnot()
        associator.associate(pcds)

    else:
        annot_list, main_data_list = create_main_and_annot_list(main_data_path)
        annot_list = annot_list[first_node : first_node + num_of_nodes]
        main_data_list = main_data_list[first_node : first_node + num_of_nodes]

        annot = AnnotatorPointCloud(annot_list)
        reflection = np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        pcd_b = PcdBuilderPointCloud(
            camera, annot, reflection
        )  # reflection is needed due to dataset (icl nuim) particularities

        for i, file in enumerate(annot_list):
            pcds.append(pcd_b.build_pcd(main_data_list[i], i))

        associator = AssociatorFront()
        associator.associate(pcds)

    max_tracks = PostProcessing.post_process(pcds)

    slam_graph = SLAMGraph()
    graph_estimated_state = slam_graph.estimate_graph(pcds, max_tracks)

    measure_error = MeasureError(ds_filename_gt, len(annot_list), num_of_nodes)
    measure_error.measure_error(first_node, first_gt_node, graph_estimated_state)

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
        "which_format",
        type=int,
        choices=[1, 2, 3],
        help="living room = 1, office = 2, point clouds = 3",
    )
    parser.add_argument(
        "first_node", type=int, help="From what node algorithm should start"
    )
    parser.add_argument(
        "first_gt_node", type=int, help="From what node gt references start"
    )
    parser.add_argument("num_of_nodes", type=int, help="Number of needed nodes")
    parser.add_argument(
        "ds_filename_gt", type=str, help="Filename of a file with gt references"
    )

    args = parser.parse_args()

    main(
        args.main_data,
        args.annot,
        args.which_format,
        args.first_node,
        args.first_gt_node,
        args.num_of_nodes,
        args.ds_filename_gt,
    )
