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
    elif transform == 'identity':
        return image
    else:
        raise ValueError("Unknown transformation type")


src_dir = '/Users/maria/Documents/GitHub/TFM/OCTA_time_series/GRAPH_matching/data/soul_longitudinal/'

for seg in os.listdir(os.path.join(src_dir, 'segmentations')):
    if not seg.endswith('.png'):
        continue

    # Segmentation_T and everything with _T is the IMAGE POST TREATMENT
    segmentation = np.array(Image.open(os.path.join(src_dir, 'segmentations', seg)))
    segmentation_T =  np.array(Image.open(os.path.join(src_dir, 'transformed_images', seg.split('.')[0] + '_1.png')))

    # Load the reference graph: seg.split('_')[0]
    edges_file = os.path.join(src_dir, 'graph_extracted_full', seg.split('.')[0] + '_full_edges.csv')
    nodes_file = os.path.join(src_dir, 'graph_extracted_full', seg.split('.')[0] + '_full_nodes.csv')
    json_dir = os.path.join(src_dir, 'graph_extracted_full', seg.split('.')[0] + '_full_graph.json')
    new_json_file = os.path.join(src_dir, 'graph_extracted_full', seg.split('.')[0] + '_full_graph_filtered.json')
    data_edges_graph, json_edges = load_data_graph(edges_file, nodes_file, json_dir, new_json_file)
    color_graph = colored_graph(os.path.join(src_dir, 'segmentations', seg.split('.')[0] + '.png'), new_json_file)

    # Load the transformed graph
    edges_file_T = os.path.join(src_dir, 'graph_extracted_full', seg.split('.')[0] + '_1_full_edges.csv')
    nodes_file_T = os.path.join(src_dir, 'graph_extracted_full', seg.split('.')[0] + '_1_full_nodes.csv')
    json_dir_T = os.path.join(src_dir, 'graph_extracted_full', seg.split('.')[0] + '_1_full_graph.json')
    new_json_file_T = os.path.join(src_dir, 'graph_extracted_full', seg.split('.')[0] + '_1_full_graph_filtered.json')
    data_edges_graph_T, json_edges_T = load_data_graph(edges_file_T, nodes_file_T, json_dir_T, new_json_file_T)
    color_graph_T = colored_graph(os.path.join(src_dir, 'transformed_images', seg.split('.')[0] + '_1.png'), new_json_file_T)

    # Get info from CSV files
    vessels_pts_graph_T, bif_pts_graph_T = get_info_graph(data_edges_graph_T, json_edges_T)
    vessels_pts_graph, bif_pts_graph = get_info_graph(data_edges_graph, json_edges)

    # Create GT
    transform_type = 'identity'
    parameters = None
    gt_1to2 = create_gt(vessels_pts_graph, color_graph, segmentation, vessels_pts_graph_T, color_graph_T, segmentation_T, inverse_transform, transform_type, parameters)

    # Save the ground truth as a matrix
    n1 = len(data_edges_graph)
    n2 = len(data_edges_graph_T)
    GT = np.zeros((n1, n2))

    for key, value in gt_1to2.items():
        GT[key, value] = 1

    
    save_gt = os.path.join(src_dir, 'gt', seg.split('.')[0] + '_1.npy')
    np.save(save_gt, GT)
    print('Done')


