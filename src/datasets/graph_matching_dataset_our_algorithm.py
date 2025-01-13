import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools.affinity_matrix_tools import extract_graph_features, build_adjacency_matrix
from tools.get_coords_and_info import get_info_graph
from visualization_tools.vvg_loader import vvg_to_df

class GraphMatchingDataset(Dataset):
    def __init__(self, src_dir, transforms_dir):
        """
        Args:
            src_dir (str): Directory where the source data is stored.            
            transforms_dir (str): Directory containing the transformed images.
        """
        self.src_dir = src_dir
        self.transforms_dir = transforms_dir

        self.indexes = []
        self.graph1_attr = []
        self.graph2_attr = []
        self.GT = []

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

            # Get the Ground Truth
            gt_dir = os.path.join(self.src_dir, 'gt', T_seg.split('.')[0] + '.npy')
            gt = np.load(gt_dir)
            gt_tensor = torch.from_numpy(gt).to(torch.float32)

            # Store all the data in lists
            self.indexes.append(T_seg.split('.')[0])
            """self.graph1_attr.append([data_edges_graph, json_edges])
            self.graph2_attr.append([data_edges_graph_T, json_edges_T])"""
            self.GT.append(gt_tensor)

    def _load_data_graph(self, edges_file, json_dir):
        """Helper function to load graph data from CSV and JSON files."""
        data_edges_graph = pd.read_csv(edges_file, delimiter=';')
        json_edges, json_nodes = vvg_to_df(json_dir)
        return data_edges_graph, json_edges

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        """Get an item by index, converting attributes to PyTorch tensors."""
        
        index = self.indexes[idx]
        gt = self.GT[idx]

        return index, gt # (graph1_edges, graph1_json), (graph2_edges, graph2_json), gt
    
