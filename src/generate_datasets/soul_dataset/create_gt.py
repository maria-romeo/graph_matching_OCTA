import numpy as np
import os
import random 
import pandas as pd
from PIL import Image

import sys
sys.path.append(os.path.abspath('../../'))
from transformation_functions import *
from tools.sanity_check_json import sanity_check_json
from visualization_tools.vvg_loader import vvg_to_df
from tools.ground_truth_generation import create_gt   
from tools.get_coords_and_info import get_info_graph
from visualization_tools.vvg_colored_graph import colored_graph

# Define the function to load the data from the CSV files
def load_data_graph(edges_file, nodes_file, json_dir, new_json_file):
    data_edges_graph = pd.read_csv(edges_file, delimiter=';')
    sanity_check_json(nodes_file, edges_file, json_dir, new_json_file)
    json_edges, json_nodes = vvg_to_df(new_json_file)
    return data_edges_graph, json_edges

# Define the inverse transformation
def inverse_transform(transform, image, parameters):
    if transform == 'translate':
        tx, ty = parameters
        return recover_translation(image, tx, ty)
    elif transform == 'rotate':
        center_x, center_y, angle = parameters
        return recover_rotation(image, center_x, center_y, angle)
    elif transform == 'elastic':
        output_def, idx = parameters
        return recover_elastic(image, output_def, idx)
    else:
        raise ValueError("Unknown transformation type")


src_dir = '/Users/maria/Documents/GitHub/TFM/OCTA_time_series/GRAPH_matching/data/soul_dataset_big/'

# Read transformations_log.csv file
log_csv_path = os.path.join(src_dir, "transformations_log.csv")
if os.path.exists(log_csv_path):
    df_transformations = pd.read_csv(log_csv_path)


for idx, row in df_transformations.iterrows():
    # Load segmentation
    seg = row['index']
    segmentation_T = np.array(Image.open(os.path.join(src_dir, 'transformed_images', seg + '.png')))
    segmentation = np.array(Image.open(os.path.join(src_dir, 'segmentations', seg.split('_')[0] + '.png')))

    # Load the reference graph: seg.split('_')[0]
    edges_file = os.path.join(src_dir, 'graph_extracted_full', seg.split('_')[0] + '_full_edges.csv')
    nodes_file = os.path.join(src_dir, 'graph_extracted_full', seg.split('_')[0] + '_full_nodes.csv')
    json_dir = os.path.join(src_dir, 'graph_extracted_full', seg.split('_')[0] + '_full_graph.json')
    new_json_file = os.path.join(src_dir, 'graph_extracted_full', seg.split('_')[0] + '_full_graph_filtered.json')
    data_edges_graph, json_edges = load_data_graph(edges_file, nodes_file, json_dir, new_json_file)
    color_graph = colored_graph(os.path.join(src_dir, 'segmentations', seg.split('_')[0] + '.png'), new_json_file)

    # Load the transformed graph
    edges_file_T = os.path.join(src_dir, 'graph_extracted_full', seg + '_full_edges.csv')
    nodes_file_T = os.path.join(src_dir, 'graph_extracted_full', seg + '_full_nodes.csv')
    json_dir_T = os.path.join(src_dir, 'graph_extracted_full', seg + '_full_graph.json')
    new_json_file_T = os.path.join(src_dir, 'graph_extracted_full', seg + '_full_graph_filtered.json')
    data_edges_graph_T, json_edges_T = load_data_graph(edges_file_T, nodes_file_T, json_dir_T, new_json_file_T)
    color_graph_T = colored_graph(os.path.join(src_dir, 'transformed_images', seg + '.png'), new_json_file_T)

    # Get info from CSV files
    vessels_pts_graph, bif_pts_graph = get_info_graph(data_edges_graph, json_edges)
    vessels_pts_graph_T, bif_pts_graph_T = get_info_graph(data_edges_graph_T, json_edges_T)

    # Apply transformations
    if row['transform_type'] == 'translate':
        parameters_trans = int(row['tx']), int(row['ty'])
        gt_1to_T = create_gt(vessels_pts_graph, color_graph, segmentation, vessels_pts_graph_T, color_graph_T, segmentation_T, inverse_transform, row['transform_type'], parameters_trans)

    elif row['transform_type'] == 'rotate':
        parameters_rot = segmentation_T.shape[0], segmentation_T.shape[1], int(row['angle'])
        gt_1to_T = create_gt(vessels_pts_graph, color_graph, segmentation, vessels_pts_graph_T, color_graph_T, segmentation_T, inverse_transform, row['transform_type'], parameters_rot)

    elif row['transform_type'] == 'elastic':
        parameters_elas = os.path.join(src_dir, 'elastic_def_matrix'), row['index']
        gt_1to_T = create_gt(vessels_pts_graph, color_graph, segmentation, vessels_pts_graph_T, color_graph_T, segmentation_T, inverse_transform, row['transform_type'], parameters_elas)

    else:
        raise ValueError("Unknown transformation type")
    
    # Save the ground truth as a matrix
    n1 = len(data_edges_graph)
    n2 = len(data_edges_graph_T)
    GT = np.zeros((n1, n2))

    for key, value in gt_1to_T.items():
        GT[key, value] = 1

    save_gt = os.path.join(src_dir, 'gt', seg + '.npy')
    np.save(save_gt, GT)