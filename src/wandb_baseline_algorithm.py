import numpy as np
import torch
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pygmtools as pygm
pygm.set_backend('pytorch')

from torch.utils.data import DataLoader

import wandb

from datasets.graph_matching_dataset_our_algorithm import GraphMatchingDataset

from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import Subset

from visualization_tools.vvg_loader import vvg_to_df
from tools.baseline_algorithm import spatial_vessel_matching
from tools.calculate_metrics import calculate_metrics

def _load_data_graph(edges_file, json_dir):
        """Helper function to load graph data from CSV and JSON files."""
        data_edges_graph = pd.read_csv(edges_file, delimiter=';')
        json_edges, json_nodes = vvg_to_df(json_dir)
        return data_edges_graph, json_edges

def idxs_from_matrix(X):
    final_idxs = []
    for i in range(len(X)):
        point1 = i  # we are matching graph 1 to graph 2!!!
        point2 = torch.argmax(X[i]).item()
        final_idxs.append({'graph1': point1, 'graph2': point2})
    return final_idxs

def idxs_gt(X):
    final_idxs = {}
    for i in range(len(X)):
        point1 = i  # we are matching graph 1 to graph 2!!!
        point2 = torch.argmax(X[i]).item()
        if X[i,point2] == 1: # otherwise there is no match
            final_idxs[point1] = point2
    return final_idxs


valid_feats1 = ['length', 'distance', 'curveness', 'volume', 'avgCrossSection', 'minRadiusAvg', 'avgRadiusAvg', 'roundnessAvg']
valid_feats2 = ['length', 'distance', 'volume', 'curveness', 'avgCrossSection', 'avgRadiusAvg']

## WANDB ##
# Define sweep config
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "mean_acc"},
    "parameters": {
        "valid_feats": {"values": [valid_feats2]},#['avgCrossSection', 'avgRadiusAvg']]}, #,'length', 'distance', 'curveness', 'volume', 'avgCrossSection', 'minRadiusAvg', 'avgRadiusAvg', 'roundnessAvg']},
        "dataset": {"values": ['soul_big']}, #, "DCP"
        "normalize_feats": {"values": [True, False]},
        "threshold_dist": {"values": [20]},
        "n_neighbors": {"values": [10, 20]},
        "factors_dist": {"values": [[4, 5, 6, 7, 10], [5, 4, 7, 6, 10], [4, 8, 4, 8, 100], [8, 4, 8, 4, 10], [8, 4, 8, 8, 100], [4, 8, 8, 8, 100], [100, 100, 100, 100, 100]]}, # [4, 4, 5, 5, 7], 
        "hungarian": {"values": [True]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="GM_baseline_ours")


i = 0

def main():
    global i 
    i +=1
    run = wandb.init()
    name = f"{i}_ours_dataset{wandb.config.dataset}"
    wandb.run.name = name
    api = wandb.Api()
    sweep = api.sweep("ge92lik/GM_baseline_ours/" + sweep_id)

    
    # Initialize the dataset
    if wandb.config.dataset == 'soul':
        dataset_name = 'soul_dataset'
    elif wandb.config.dataset == 'soul_big':
        dataset_name = 'soul_dataset_big'
    src_dir = f'../data/{dataset_name}/'
    transforms_dir = os.path.join(src_dir, 'transformed_images')

    # Initialize the dataset
    dataset = GraphMatchingDataset(src_dir=src_dir,  
                                transforms_dir=transforms_dir)

    # Group images by subject
    subject_to_images = defaultdict(list)
    for index in dataset.indexes:
        subject = index.split('_')[0]  # Extract the subject number
        subject_to_images[subject].append(index)

    # Convert grouped data to a list of subject groups for splitting
    subject_groups = list(subject_to_images.values())

    train_subjects, tmp_subjects = train_test_split(subject_groups, test_size=0.40, random_state=42)
    val_subjects, test_subjects = train_test_split(tmp_subjects, test_size=0.50, random_state=42)

    # Flatten lists of images per split
    train_idxs = [img for subject in train_subjects for img in subject]
    val_idxs = [img for subject in val_subjects for img in subject]
    test_idxs = [img for subject in test_subjects for img in subject]

    # Combine train and validation indexes to create a larger training set
    train_val_idxs = train_idxs + val_idxs

    # Convert the indexes back to actual dataset indices
    train_val_set = Subset(dataset, [dataset.indexes.index(idx) for idx in train_val_idxs])  # Combined train+val
    test_set = Subset(dataset, [dataset.indexes.index(idx) for idx in test_idxs])

    # Create DataLoaders for each set
    train_val_loader = DataLoader(train_val_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # Iterate over DataLoader
    for batch_idx, (index, gt) in enumerate(train_val_loader):

        T_seg = index[0]
        seg = T_seg.split('_')[0]

        edges_file = os.path.join(src_dir, 'graph_extracted_full', seg + '_full_edges.csv')
        json_dir = os.path.join(src_dir, 'graph_extracted_full', seg + '_full_graph_filtered.json')
        data_edges_graph, json_edges = _load_data_graph(edges_file, json_dir)

        # Load the transformed img graph data
        edges_file_T = os.path.join(src_dir, 'graph_extracted_full', T_seg + '_full_edges.csv')
        json_dir_T = os.path.join(src_dir, 'graph_extracted_full', T_seg + '_full_graph_filtered.json')
        data_edges_graph_T, json_edges_T = _load_data_graph(edges_file_T, json_dir_T)

        # Normalize valid features in data_edges_graph_T
        if wandb.config.normalize_feats:
            scaler = MinMaxScaler()
            data_edges_graph[wandb.config.valid_feats] = scaler.fit_transform(data_edges_graph[wandb.config.valid_feats])
            data_edges_graph_T[wandb.config.valid_feats] = scaler.fit_transform(data_edges_graph_T[wandb.config.valid_feats])
             

        dict_data_edges = {'graph1': data_edges_graph, 'graph2': data_edges_graph_T}
        dict_json_edges = {'graph1': json_edges, 'graph2': json_edges_T}


        # MATCHING
        matching_set = spatial_vessel_matching(dict_data_edges, dict_json_edges, wandb.config.valid_feats, wandb.config.threshold_dist, wandb.config.n_neighbors, wandb.config.factors_dist)
        matrix_hung, final_coords_graphs, final_info_graphs, final_idxs_graphs = matching_set

        if wandb.config.hungarian:
            X = pygm.hungarian(torch.from_numpy(matrix_hung))
            final_idxs = idxs_from_matrix(X)
        else:
            final_idxs = final_idxs_graphs

        # Get the GT indexes
        idx_gt = idxs_gt(gt[0])

        # EVALUATION
        acc, prec, recall, f1 = calculate_metrics(final_idxs, idx_gt, json_edges_T)

        accuracies.append(acc) 
        precisions.append(prec)
        recalls.append(recall)
        f1s.append(f1)

    mean_acc = torch.tensor(accuracies).mean()
    mean_prec = torch.tensor(precisions).mean()
    mean_recall = torch.tensor(recalls).mean()
    mean_f1 = torch.tensor(f1s).mean()
    std_acc = torch.tensor(accuracies).std()
    std_prec = torch.tensor(precisions).std()
    std_recall = torch.tensor(recalls).std()
    std_f1 = torch.tensor(f1s).std()
    
    wandb.log({"acc": mean_acc, 
               "std_acc": std_acc,
                "prec": mean_prec, 
                "std_prec": std_prec,
                "recall": mean_recall, 
                "std_recall": std_recall,
                "f1": mean_f1,
                "std_f1": std_f1
    })
wandb.agent(sweep_id, function=main, count=10000)
