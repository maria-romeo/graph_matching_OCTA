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

def matrix_from_idxs(final_idxs, matrix_shape):
    """
    Construct a matrix from a dictionary of matched indices based on the shape of matrix_hung.

    Parameters:
    - final_idxs (list of dicts): Each dictionary contains 'graph1' and 'graph2' keys indicating matched nodes.
    - matrix_shape (torch.Tensor): The matrix whose shape will be used for the output matrix.

    Returns:
    - X (torch.Tensor): The constructed matrix of the same shape as matrix_hung with 1s indicating matches.
    """

    # Initialize a matrix of zeros with the same shape as matrix_hung
    X = np.zeros_like(matrix_shape)

    # Iterate over the final_idxs and set the corresponding entries in the matrix to 1
    for match in final_idxs:
        point1 = match['graph1']
        point2 = match['graph2']
        X[point1, point2] = 1

    return X

def idxs_gt(X):
    final_idxs = {}
    for i in range(len(X)):
        point1 = i  # we are matching graph 1 to graph 2!!!
        point2 = torch.argmax(X[i]).item()
        if X[i,point2] == 1: # otherwise there is no match
            final_idxs[point1] = point2
    return final_idxs


valid_feats1 = ['length', 'distance', 'curveness', 'volume', 'avgCrossSection', 'minRadiusAvg', 'avgRadiusAvg', 'roundnessAvg']
valid_feats2 = ['length', 'distance', 'volume', 'avgCrossSection', 'avgRadiusAvg']

## WANDB ##
# Define sweep config
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "mean_acc"},
    "parameters": {
        "valid_feats": {"values": [valid_feats2]}, #,'length', 'distance', 'curveness', 'volume', 'avgCrossSection', 'minRadiusAvg', 'avgRadiusAvg', 'roundnessAvg']},
        "dataset": {"values": ['soul_big']}, #, "DCP"
        "normalize_feats": {"values": [True]},# True, False
        "threshold_dist": {"values": [100]}, # 20, 30, 50, 70, 100
        "n_neighbors": {"values": [20]}, # 5, 10, 15, 20, 25, 30
        "factors_dist": {"values": [[4, 4, 5, 5, 7]]},#, [4, 5, 6, 7, 10], [5, 4, 7, 6, 10], [4, 8, 4, 8, 100], [8, 4, 8, 4, 10], [8, 4, 8, 8, 100], [4, 8, 8, 8, 100], [100, 100, 100, 100, 100]]},
        "hungarian": {"values": [True]}, # True, False
    },
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="GM_test")


i = 0

def main():
    global i 
    i +=1
    run = wandb.init()
    name = f"{i}_ours_dataset{wandb.config.dataset}"
    wandb.run.name = name
    api = wandb.Api()
    sweep = api.sweep("ge92lik/GM_test/" + sweep_id)

    
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
    test_idxs = [img for subject in test_subjects for img in subject]
    # Convert the indexes back to actual dataset indices
    test_set = Subset(dataset, [dataset.indexes.index(idx) for idx in test_idxs])

    # Create DataLoaders for each set
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # Store the evaluation metrics for every sample
    records = []
    # Create dirs to store results
    if not os.path.exists('GM_results'):
        os.makedirs('GM_results')
    if not os.path.exists(f'GM_results/baseline/baseline_matrix_results'):
        os.makedirs(f'GM_results/baseline/baseline_matrix_results')
    x_pred_dir = f'GM_results/baseline/baseline_matrix_results'

    # Iterate over DataLoader
    for batch_idx, (index, gt) in enumerate(test_loader):

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

        # Collect the results into a list of dictionaries
        records.append({
            "index": index.item() if isinstance(index, torch.Tensor) else index,
            "accuracy": acc,
            "precision": prec,
            "recall": recall,
            "f1": f1
        })

        # Save X_pred matrix
        if wandb.config.hungarian:
            X_pred_np = X.detach().cpu().numpy()
        else:
            X_pred_np = matrix_from_idxs(final_idxs, matrix_hung)

        x_pred_filename = os.path.join(x_pred_dir, f'{index[0]}.npy')
        np.save(x_pred_filename, X_pred_np)

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

    # STORE METRICS, MODEL CONFIG, AND RESULTS

    # Save the model configuration
    # Write model and features to a text file
    with open(f'GM_results/baseline/baseline_config_info.txt', 'a') as config_file:
        config_file.write(f"Model: Baseline Algorithm\n")
        config_file.write(f"Dataset: {wandb.config.dataset}\n")
        config_file.write(f"Features: {wandb.config.valid_feats}\n")
        config_file.write(f"Normalize Features: {wandb.config.normalize_feats}\n") 
        config_file.write(f"Threshold Distance: {wandb.config.threshold_dist}\n")
        config_file.write(f"Number of Neighbors: {wandb.config.n_neighbors}\n")
        config_file.write(f"Factors Distance: {wandb.config.factors_dist}\n")
        config_file.write(f"Hungarian: {wandb.config.hungarian}\n")
        config_file.write("\n")

    # Save the evaluation metrics
    results_df = pd.DataFrame(records)
    results_df.to_csv(f'GM_results/baseline/baseline_evaluation_metrics.csv', index=False)

    # Save the evaluation metrics
    with open(f'GM_results/baseline/baseline_avge_evaluation_metrics.txt', 'a') as metrics_file:
        metrics_file.write(f"Mean Test Accuracy: {mean_acc}, Mean Test Precision: {mean_prec}, Mean Test Recall: {mean_recall}, Mean Test F1 Score: {mean_f1}\n")
        metrics_file.write(f"Std Test Accuracy: {std_acc}, Std Test Precision: {std_prec}, Std Test Recall: {std_recall}, Std Test F1 Score: {std_f1}\n")
        metrics_file.write("\n")
    

wandb.agent(sweep_id, function=main, count=400)