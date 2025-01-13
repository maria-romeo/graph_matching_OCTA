import wandb
from torch.utils.data import DataLoader
from datasets.graph_matching_dataset import GraphMatchingDataset
import os
import pygmtools as pygm
import torch
from tools.calculate_metrics import calculate_metrics
import pandas as pd
import functools
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import Subset
import sys
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, '..'))
from visualization_tools.vvg_loader import vvg_to_df

def _load_data_graph(edges_file, json_dir):
        """Helper function to load graph data from CSV and JSON files."""
        data_edges_graph = pd.read_csv(edges_file, delimiter=';')
        json_edges, json_nodes = vvg_to_df(json_dir)
        return data_edges_graph, json_edges

def idxs_gt(X):
    final_idxs = {}
    for i in range(len(X)):
        point1 = i  # we are matching graph 1 to graph 2!!!
        point2 = torch.argmax(X[i]).item()
        if X[i,point2] == 1: # otherwise there is no match
            final_idxs[point1] = point2
    return final_idxs

def idxs_pred_mat(X):
    final_idxs = []
    for i in range((X.shape[0])):
        point1 = i  # we are matching graph 1 to graph 2!!!
        point2 = torch.argmax(X[i]).item()
        final_idxs.append({'graph1': point1, 'graph2': point2})
    return final_idxs

def normalize_features(feat):
    """
    Normalize node or edge MATRIX features depending on the shape of the input tensor.
    """
    
    if len(feat.shape) == 3:  # Node features: [batch_size, num_nodes, num_features]
        # Normalize across the node dimension (dim=1), keep batch and feature dims intact
        mean = feat.mean(dim=(0, 1), keepdim=True)  # Mean across nodes for each feature
        std = feat.std(dim=(0, 1), keepdim=True)    # Std across nodes for each feature
    
    elif len(feat.shape) == 4:  # Edge MATRIX features: [batch_size, num_nodes, num_nodes, num_edge_features]
        # Normalize across the node-to-node edge dimensions (dim=0, 1, 2), keep edge feature dim intact
        mean = feat.mean(dim=(0, 1, 2), keepdim=True)  # Mean across all edges
        std = feat.std(dim=(0, 1, 2), keepdim=True)    # Std across all edges
    
    # Normalize the features
    normalized_feat = (feat - mean) / (std + 1e-5)
    
    return normalized_feat


valid_feats1 = ['length', 'distance', 'curveness', 'volume', 'avgCrossSection', 'minRadiusAvg', 'avgRadiusAvg', 'roundnessAvg']
valid_feats2 = ['length', 'distance', 'volume', 'avgCrossSection', 'avgRadiusAvg']

## WANDB ##
# Define sweep config
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "mean_acc"},
    "parameters": {
        "model": {"values": ['ipfp', 'rrwm', 'sm']},
        "dataset": {"values": ['soul_big']}, #'soul', 
        "valid_feats": {"values": [['length'], ['distance'], ['curveness'], ['volume'], ['avgCrossSection'], ['minRadiusAvg'], ['avgRadiusAvg'], ['roundnessAvg']]},        
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="GM_pygm_classic_solvers")


i = 0
def main():
    global i 
    i +=1
    run = wandb.init()
    name = f"{i}_{wandb.config.model}_dataset{wandb.config.dataset}"
    wandb.run.name = name
    api = wandb.Api()
    sweep = api.sweep("ge92lik/GM_pygm_classic_solvers/" + sweep_id)

    if wandb.config.dataset == 'soul':
        dataset_name = 'soul_dataset'
    elif wandb.config.dataset == 'soul_big':
        dataset_name = 'soul_dataset_big'
    src_dir = f'../data/{dataset_name}/'
    transforms_dir = os.path.join(src_dir, 'transformed_images')

    # Initialize the dataset
    dataset = GraphMatchingDataset(src_dir=src_dir, 
                                valid_feats=wandb.config.valid_feats, 
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


    pygm.set_backend('pytorch')

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # Iterate over DataLoader
    for batch_idx, (index, graph1, graph2, gt, data_files) in enumerate(train_val_loader):

        A, node_feat, edge_feat, edge_feat_matrix, conn, n, ne = graph1
        A_T, node_feat_T, edge_feat_T, edge_feat_matrix_T, conn_T, n_T, ne_T = graph2

        # Normalize the features
        node_feat = normalize_features(node_feat)
        node_feat_T = normalize_features(node_feat_T)
        edge_feat = normalize_features(edge_feat)
        edge_feat_T = normalize_features(edge_feat_T)
        edge_feat_matrix = normalize_features(edge_feat_matrix)
        edge_feat_matrix_T = normalize_features(edge_feat_matrix_T)
        
        # Step 4: Build the affinity matrix
        # Build the affinity matrix
        conn_k = conn.long()
        conn_T_k = conn_T.long()
        gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) 
        K = pygm.utils.build_aff_mat(node_feat, edge_feat, conn_k, node_feat_T, edge_feat_T, conn_T_k, n, ne, n_T, ne_T, edge_aff_fn=gaussian_aff)


        # Step 5: Run the iPCA-GM algorithm with pretrained model
        if wandb.config.model == 'ipfp':
            X_T = pygm.ipfp(K, n, n_T)
        elif wandb.config.model == 'rrwm':
            X_T = pygm.rrwm(K, n, n_T)
        elif wandb.config.model == 'sm':
            X_T = pygm.sm(K, n, n_T)


        # Calculate accuracy metrics
        X_pred = pygm.hungarian(X_T) # Apply hungarian after calculating the loss bc its not differentiable
        if len(X_pred.shape) == 3:
            X_pred = X_pred.squeeze(0)
        idxs_X_T = idxs_pred_mat(X_pred)
        idx_gt = idxs_gt(gt[0])
        edges_file_T, json_dir_T = data_files
        data_edges_graph_T, json_edges_T = _load_data_graph(edges_file_T[0], json_dir_T[0])
        acc, prec, recall, f1 = calculate_metrics(idxs_X_T, idx_gt, json_edges_T)
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
