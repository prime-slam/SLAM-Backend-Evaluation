import argparse
import os

import config
from associators.AssociatorAnnot import AssociatorAnnot
from associators.AssociatorFront import AssociatorFront
from MeasureError import MeasureError
from pcdBuilders.PcdBuilderLiving import PcdBuilderLiving
from pcdBuilders.PcdBuilderOffice import PcdBuilderOffice
from pcdBuilders.PcdBuilderPointCloud import PcdBuilderPointcloud
from PostProcessing import PostProcessing
from SLAMGraph import SLAMGraph
from Visualisation import Visualisation


def main(main_data: str,
         annot: str,
         which_format: int,
         first_node: int,
         first_gt_node: int,
         num_of_nodes: int,
         ds_filename_gt: str,
         file_name_estimated: str,
         file_name_gt: str):

    camera = config.CAMERA_ICL
    pcds = []
    annot_list = []
    main_data_list = []

    if which_format == 1 or which_format == 2:
        depth = os.listdir(main_data)
        depth = sorted(depth, key=lambda x: int(x[:-4]))
        main_data_list = list(map(lambda x: os.path.join(main_data, x), depth))

        colors_orig = os.listdir(annot)
        colors_orig = sorted(colors_orig, key=lambda x: int(x[:-4]))
        annot_list = list(map(lambda x: os.path.join(annot, x), colors_orig))

        if which_format == 1:
            pcd_b = PcdBuilderLiving(camera, annot_list)
        else:
            pcd_b = PcdBuilderOffice(camera, annot_list)

        for i, image in enumerate(annot_list):
            pcds.append(pcd_b.build_pcd(i, main_data_list))

        associator = AssociatorAnnot(pcds, color_to_indx={})
        associator.associate()

    else:
        folders = os.listdir(main_data)

        for j, folder in enumerate(folders):
            cur_path = os.path.join(main_data, folder)
            npy, pcd = os.listdir(cur_path)

            cur_path_to_annot = os.path.join(cur_path, npy)
            cur_path_to_main_data = os.path.join(cur_path, pcd)

            annot_list.append(cur_path_to_annot)
            main_data_list.append(cur_path_to_main_data)

        pcd_b = PcdBuilderPointcloud(camera, annot_list)

        for i, file in annot_list:
            pcds.append(pcd_b.build_pcd(i, annot_list))

        associator = AssociatorFront(pcds)
        associator.associate()

    post_processing = PostProcessing(pcds)
    post_processing.post_process()

    for i, pcd in pcds:
        if i < first_node or i > first_node + num_of_nodes:
            pcds.remove(pcd)

    slam_graph = SLAMGraph()
    graph_estimated_state = slam_graph.estimate_the_graph(pcds)

    measure_error = MeasureError(first_node,
                                 first_gt_node,
                                 len(pcds),
                                 len(annot_list),
                                 graph_estimated_state,
                                 ds_filename_gt,
                                 file_name_estimated,
                                 file_name_gt)
    measure_error.measure_error()

    visual = Visualisation(graph_estimated_state, num_of_nodes, pcds)
    visual.visualisation()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build a trajectory')
    parser.add_argument('main_data', type=str, help='Directory where color images are stored')
    parser.add_argument('annot', type=str, help='Directory where color images are stored')
    parser.add_argument('which_format', type=int, help='Directory where color images are stored')
    parser.add_argument('first_node', type=int, help='Directory where color images are stored')
    parser.add_argument('first_gt_node', type=int, help='Directory where color images are stored')
    parser.add_argument('num_of_nodes', type=int, help='Directory where color images are stored')
    parser.add_argument('ds_filename_gt', type=str, help='Directory where color images are stored')
    parser.add_argument('file_name_estimated', type=str, help='Directory where color images are stored')
    parser.add_argument('file_name_gt', type=str, help='Directory where color images are stored')


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