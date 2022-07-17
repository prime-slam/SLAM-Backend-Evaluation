import argparse
import os

import config
from SLAMGraph import SLAMGraph
from annotators.AnnotaatorImage import AnnotatorImage
from annotators.AnnotatorPointCloud import AnnotatorPointCloud
from associators.AssociatorAnnot import AssociatorAnnot
from associators.AssociatorFront import AssociatorFront
from measurements.MeasureError import MeasureError
from pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from pcdBuilders.PcdBuilderOffice import PcdBuilderOffice
from pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointcloud
from PostProcessing import PostProcessing
from Visualisation import Visualisation


def create_main_list_image(main_data_path: str):
    depth = os.listdir(main_data_path)
    depth = sorted(depth, key=lambda x: int(x[:-4]))
    main_data_list = list(map(lambda x: os.path.join(main_data_path, x), depth))
    return main_data_list


def create_annot_list_image(annot_path):
    colors_orig = os.listdir(annot_path)
    colors_orig = sorted(colors_orig, key=lambda x: int(x[:-4]))
    annot_list = list(map(lambda x: os.path.join(annot_path, x), colors_orig))
    return annot_list


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


def main(main_data_path: str,
         annot_path: str,
         which_format: int,
         first_node: int,
         first_gt_node: int,
         num_of_nodes: int,
         ds_filename_gt: str,
         file_name_estimated: str,
         file_name_gt: str):

    camera = config.CAMERA_ICL
    pcds = []

    if which_format == 1 or which_format == 2:

        main_data_list = create_main_list_image(main_data_path)[first_node: first_node + num_of_nodes]
        annot_list = create_annot_list_image(annot_path)[first_node: first_node + num_of_nodes]

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
        annot = AnnotatorPointCloud(annot_list)
        pcd_b = PcdBuilderPointcloud(camera, annot)

        for i, file in enumerate(annot_list):
            pcds.append(pcd_b.build_pcd(main_data_list[i], i))

        associator = AssociatorFront()
        associator.associate(pcds)

    post_processing = PostProcessing()
    max_tracks = post_processing.post_process(pcds)

    slam_graph = SLAMGraph()
    graph_estimated_state = slam_graph.estimate_the_graph(pcds, num_of_nodes, first_node,  max_tracks)

    measure_error = MeasureError(ds_filename_gt,
                                 len(annot_list))
    measure_error.measure_error(first_node,
                                first_gt_node,
                                graph_estimated_state,
                                file_name_estimated,
                                file_name_gt)

    visualisation = Visualisation(graph_estimated_state)
    visualisation.visualisation(pcds, graph_estimated_state)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build a trajectory')
    parser.add_argument('main_data', type=str, help='Directory where main information files are stored')
    parser.add_argument('annot', type=str, help='Directory where color images are stored')
    parser.add_argument('which_format', type=int, choices=[1, 2, 3], help='living room = 1, office = 2, point clouds = 3')
    parser.add_argument('first_node', type=int, help='from what node algorithm should start')
    parser.add_argument('first_gt_node', type=int, help='From what node gt references start')
    parser.add_argument('num_of_nodes', type=int, help='Directory where color images are stored')
    parser.add_argument('ds_filename_gt', type=str, help='Filename of a file with gt references')
    parser.add_argument('file_name_estimated', type=str, help='Where to write estimated quaternions')
    parser.add_argument('file_name_gt', type=str, help='Where to write needed gt references')

    args = parser.parse_args()

    main(args.main_data,
         args.annot,
         args.which_format,
         args.first_node,
         args.first_gt_node,
         args.num_of_nodes,
         args.ds_filename_gt,
         args.file_name_estimated,
         args.file_name_gt)
