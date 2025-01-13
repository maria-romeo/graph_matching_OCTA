import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import pygmtools as pygm
import functools
import os

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools.get_coords_and_info import get_info_graph
from tools.affinity_matrix_tools import extract_graph_features
from visualization_tools.vvg_loader import vvg_to_df

class GraphMatchingDataset(Dataset):
    def __init__(self, src_dir, transforms_dir, valid_feats):
        """
        Args:
            src_dir (str): Directory where the source data is stored.            
            transforms_dir (str): Directory containing the transformed images.
            valid_feats (list): List of valid features to extract from graphs.
        """
        self.src_dir = src_dir
        self.valid_feats = valid_feats
        self.transforms_dir = transforms_dir

        self.indexes = []
        self.graph1_attr = []
        self.graph2_attr = []
        self.GT = []
        self.data_files = []

        self._load_data()

    def _load_data(self):
        """Internal function to load the graph data from the specified directories."""
        for T_seg in os.listdir(self.transforms_dir):
            if not T_seg.endswith('.png'):
                continue

            # Load the reference img graph data
            seg = T_seg.split('_')[0]
            edges_file = os.path.join(self.src_dir, 'graph_extracted_full', seg + '_full_edges.csv')
            json_dir = os.path.join(self.src_dir, 'graph_extracted_full', seg + '_full_graph_filtered.json')
            data_edges_graph, json_edges = self._load_data_graph(edges_file, json_dir)

            # Load the transformed img graph data
            edges_file_T = os.path.join(self.src_dir, 'graph_extracted_full', T_seg.split('.')[0] + '_full_edges.csv')
            json_dir_T = os.path.join(self.src_dir, 'graph_extracted_full', T_seg.split('.')[0] + '_full_graph_filtered.json')
            data_edges_graph_T, json_edges_T = self._load_data_graph(edges_file_T, json_dir_T)

            # Get info from CSV files
            vessels_pts_graph, bif_pts_graph = get_info_graph(data_edges_graph, json_edges)
            vessels_pts_graph_T, bif_pts_graph_T = get_info_graph(data_edges_graph_T, json_edges_T)

            # Get the graph attributes
            n, ne = len(vessels_pts_graph), len(bif_pts_graph)
            A, node_feat, edge_feat, edge_feat_matrix, conn = extract_graph_features(vessels_pts_graph, bif_pts_graph, self.valid_feats, n)

            n_T, ne_T = len(vessels_pts_graph_T), len(bif_pts_graph_T)
            A_T, node_feat_T, edge_feat_T, edge_feat_matrix_T, conn_T = extract_graph_features(vessels_pts_graph_T, bif_pts_graph_T, self.valid_feats, n_T)

            # Normalize the features
            node_feat = self.normalize_features(node_feat)
            node_feat_T = self.normalize_features(node_feat_T)
            edge_feat = self.normalize_features(edge_feat)
            edge_feat_T = self.normalize_features(edge_feat_T)

            # Convert graph attributes to PyTorch tensors
            graph1_tensors = [x.clone().detach().to(torch.float32) if isinstance(x, torch.Tensor) else x for x in [A, node_feat, edge_feat, edge_feat_matrix, conn, n, ne]]
            graph2_tensors = [x.clone().detach().to(torch.float32) if isinstance(x, torch.Tensor) else x for x in [A_T, node_feat_T, edge_feat_T, edge_feat_matrix_T, conn_T, n_T, ne_T]]

            # Get the Ground Truth
            gt_dir = os.path.join(self.src_dir, 'gt', T_seg.split('.')[0] + '.npy')
            gt = np.load(gt_dir)
            gt_tensor = torch.from_numpy(gt).to(torch.float32)

            # Store all the data in lists
            self.indexes.append(T_seg.split('.')[0])
            self.graph1_attr.append(graph1_tensors)
            self.graph2_attr.append(graph2_tensors)
            self.GT.append(gt_tensor)
            self.data_files.append([edges_file_T, json_dir_T])

    def _load_data_graph(self, edges_file, json_dir):
        """Helper function to load graph data from CSV and JSON files."""
        data_edges_graph = pd.read_csv(edges_file, delimiter=';')
        json_edges, json_nodes = vvg_to_df(json_dir)
        return data_edges_graph, json_edges
    
    def normalize_features(self, node_feat):
        # Compute mean and std across the batch and node dimensions (dim=0 and dim=1)
        mean = node_feat.mean(dim=(0, 1), keepdim=True)  # Mean across batch and nodes
        std = node_feat.std(dim=(0, 1), keepdim=True)    # Std across batch and nodes

        # Normalize the features
        normalized_feat = (node_feat - mean) / (std + 1e-5)
        
        return normalized_feat

    def __len__(self):
        return len(self.graph1_attr)

    def __getitem__(self, idx):
        """Get an item by index, converting attributes to PyTorch tensors."""
        index = self.indexes[idx]
        graph1 = self.graph1_attr[idx]
        graph2 = self.graph2_attr[idx]
        gt = self.GT[idx]
        data_file = self.data_files[idx]

        return index, tuple(graph1), tuple(graph2), gt, tuple(data_file)