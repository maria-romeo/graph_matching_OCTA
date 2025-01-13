import torch
import pygmtools as pygm
import pandas as pd
from tools.calculate_metrics import calculate_metrics
import functools
import torch.nn.functional as F
from visualization_tools.vvg_loader import vvg_to_df
import os
import numpy as np

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

# TRAINING LOOP

def train(train_loader, net, model, optimizer, device):

    net.train()

    epoch_loss = 0
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    # Iterate over DataLoader
    for batch_idx, (index, graph1, graph2, gt, data_files) in enumerate(train_loader):

        A, node_feat, edge_feat, edge_feat_matrix, conn, n, ne = [x.to(device) for x in graph1]
        A_T, node_feat_T, edge_feat_T, edge_feat_matrix_T, conn_T, n_T, ne_T = [x.to(device) for x in graph2]
        gt = gt.to(device)

        # Normalize the features
        node_feat = normalize_features(node_feat)
        node_feat_T = normalize_features(node_feat_T)
        edge_feat = normalize_features(edge_feat)
        edge_feat_T = normalize_features(edge_feat_T)
        
    
        if model == 'ipca_gm' or model == 'pca_gm' or model == 'cie':
            # Calculate the padding needed to make d = 1024 (so we can use the preatrained weights from 'voc')
            padding_size = 1024 - node_feat.shape[2]  # The number of zeros to add to the feature dimension
            # Apply padding (pad adds dimensions at the end by default, so this works for the last dimension)
            feat1_padded = F.pad(node_feat, (0, padding_size))  # Pad the last dimension
            feat2_padded = F.pad(node_feat_T, (0, padding_size))  # Pad the last dimension
        elif model == 'ngm':
            # Build the affinity matrix
            conn_k = conn.long()
            conn_T_k = conn_T.long()
            gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) 
            K = pygm.utils.build_aff_mat(node_feat, None, conn_k, node_feat_T, None, conn_T_k, n, ne, n_T, ne_T)#, edge_aff_fn=gaussian_aff)


        # Step 5: Run the iPCA-GM algorithm with pretrained model
        if model == 'ipca_gm':
            X_T = pygm.ipca_gm(feat1_padded, feat2_padded, A, A_T, n1=n, n2=n_T, network=net)
        elif model == 'pca_gm':
            X_T = pygm.pca_gm(feat1_padded, feat2_padded, A, A_T, n1=n, n2=n_T, network=net)
        elif model == 'ngm':
            X_T = pygm.ngm(K, n, n_T, network=net)
        elif model == 'cie':
            X_T = pygm.cie(feat1_padded, feat2_padded, A, A_T, edge_feat_matrix, edge_feat_matrix_T, n, n_T, network=net)

        # Calculate the loss            
        loss = pygm.utils.permutation_loss(X_T, gt)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss for this epoch
        epoch_loss += loss.item()

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

        print(f'Batch {batch_idx} - Loss: {loss.item()} - Acc: {acc} - Prec: {prec} - Recall: {recall} - F1: {f1}')
    

    # Store average loss for this epoch
    avg_loss = epoch_loss / len(train_loader)

    return net, avg_loss, accuracies, precisions, recalls, f1s
    
    



#  Validation Loop (Added)
def validate(val_loader, net, model, device):

    net.eval()  # Set the network to evaluation mode

    val_loss = 0
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_idx, (index, graph1, graph2, gt, data_files) in enumerate(val_loader):
            
            # Move data to device
            A, node_feat, edge_feat, edge_feat_matrix, conn, n, ne = [x.to(device) for x in graph1]
            A_T, node_feat_T, edge_feat_T, edge_feat_matrix_T, conn_T, n_T, ne_T = [x.to(device) for x in graph2]
            gt = gt.to(device)

            # Normalize the features (same as training loop)
            node_feat = normalize_features(node_feat)
            node_feat_T = normalize_features(node_feat_T)
            edge_feat = normalize_features(edge_feat)
            edge_feat_T = normalize_features(edge_feat_T)

            if model == 'ipca_gm' or model == 'pca_gm' or model == 'cie':
                # Calculate the padding needed to make d = 1024 (so we can use the preatrained weights from 'voc')
                padding_size = 1024 - node_feat.shape[2]  # The number of zeros to add to the feature dimension
                # Apply padding (pad adds dimensions at the end by default, so this works for the last dimension)
                feat1_padded = F.pad(node_feat, (0, padding_size))  # Pad the last dimension
                feat2_padded = F.pad(node_feat_T, (0, padding_size))  # Pad the last dimension
            elif model == 'ngm':
                # Build the affinity matrix
                conn_k = conn.long()
                conn_T_k = conn_T.long()
                gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) 
                K = pygm.utils.build_aff_mat(node_feat, None, conn_k, node_feat_T, None, conn_T_k, n, ne, n_T, ne_T, edge_aff_fn=gaussian_aff)


            # Run the model (same as training loop)
            if model == 'ipca_gm':
                X_T = pygm.ipca_gm(feat1_padded, feat2_padded, A, A_T, n1=n, n2=n_T, network=net)
            elif model == 'pca_gm':
                X_T = pygm.pca_gm(feat1_padded, feat2_padded, A, A_T, n1=n, n2=n_T, network=net)
            elif model == 'ngm':
                X_T = pygm.ngm(K, n, n_T, network=net)
            elif model == 'cie':
                X_T = pygm.cie(feat1_padded, feat2_padded, A, A_T, edge_feat_matrix, edge_feat_matrix_T, n, n_T, network=net)

            # Calculate the loss
            loss = pygm.utils.permutation_loss(X_T, gt)
            val_loss += loss.item()

            # Calculate metrics (accuracy, precision, recall, f1)
            X_pred = pygm.hungarian(X_T)
            if len(X_pred.shape) == 3:
                X_pred = X_pred.squeeze(0)
            idxs_X_T = idxs_pred_mat(X_pred)
            idx_gt = idxs_gt(gt[0])
            edges_file_T, json_dir_T = data_files
            data_edges_graph_T, json_edges_T = _load_data_graph(edges_file_T[0], json_dir_T[0])
            acc, prec, recall, f1 = calculate_metrics(idxs_X_T, idx_gt, json_edges_T)
            val_accuracies.append(acc)
            val_precisions.append(prec)
            val_recalls.append(recall)
            val_f1s.append(f1)

    avg_val_loss = val_loss / len(val_loader)
    
    return net, avg_val_loss, val_accuracies, val_precisions, val_recalls, val_f1s

#  Validation Loop (Added)
def evaluate(test_loader, net, model, device, output_dir):

    net.eval()  # Set the network to evaluation mode

    test_loss = 0
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    # Store the evaluation metrics for every sample
    records = []
    # Create dirs to store results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f'{output_dir}/pygm_neural/{model}_matrix_results'):
        os.makedirs(f'{output_dir}/pygm_neural/{model}_matrix_results')
    x_pred_dir = f'{output_dir}/pygm_neural/{model}_matrix_results'

    
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_idx, (index, graph1, graph2, gt, data_files) in enumerate(test_loader):
            
            # Move data to device
            A, node_feat, edge_feat, edge_feat_matrix, conn, n, ne = [x.to(device) for x in graph1]
            A_T, node_feat_T, edge_feat_T, edge_feat_matrix_T, conn_T, n_T, ne_T = [x.to(device) for x in graph2]
            gt = gt.to(device)
            
            # Normalize the features (same as training loop)
            node_feat = normalize_features(node_feat)
            node_feat_T = normalize_features(node_feat_T)
            edge_feat = normalize_features(edge_feat)
            edge_feat_T = normalize_features(edge_feat_T)

            if model == 'ipca_gm' or model == 'pca_gm' or model == 'cie':
                # Calculate the padding needed to make d = 1024 (so we can use the preatrained weights from 'voc')
                padding_size = 1024 - node_feat.shape[2]  # The number of zeros to add to the feature dimension
                # Apply padding (pad adds dimensions at the end by default, so this works for the last dimension)
                feat1_padded = F.pad(node_feat, (0, padding_size))  # Pad the last dimension
                feat2_padded = F.pad(node_feat_T, (0, padding_size))  # Pad the last dimension
            elif model == 'ngm':
                # Build the affinity matrix
                conn_k = conn.long()
                conn_T_k = conn_T.long()
                gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) 
                K = pygm.utils.build_aff_mat(node_feat, None, conn_k, node_feat_T, None, conn_T_k, n, ne, n_T, ne_T, edge_aff_fn=gaussian_aff)


            # Run the model (same as training loop)
            if model == 'ipca_gm':
                X_T = pygm.ipca_gm(feat1_padded, feat2_padded, A, A_T, n1=n, n2=n_T, network=net)
            elif model == 'pca_gm':
                X_T = pygm.pca_gm(feat1_padded, feat2_padded, A, A_T, n1=n, n2=n_T, network=net)
            elif model == 'ngm':
                X_T = pygm.ngm(K, n, n_T, network=net)
            elif model == 'cie':
                X_T = pygm.cie(feat1_padded, feat2_padded, A, A_T, edge_feat_matrix, edge_feat_matrix_T, n, n_T, network=net)

            print(X_T.shape)
            print(gt.shape)
            # Calculate the loss
            loss = pygm.utils.permutation_loss(X_T, gt)
            test_loss += loss.item()

            # Calculate metrics (accuracy, precision, recall, f1)
            X_pred = pygm.hungarian(X_T)
            if len(X_pred.shape) == 3:
                X_pred = X_pred.squeeze(0)
            idxs_X_T = idxs_pred_mat(X_pred)
            idx_gt = idxs_gt(gt[0])
            edges_file_T, json_dir_T = data_files
            data_edges_graph_T, json_edges_T = _load_data_graph(edges_file_T[0], json_dir_T[0])
            acc, prec, recall, f1 = calculate_metrics(idxs_X_T, idx_gt, json_edges_T)
            test_accuracies.append(acc)
            test_precisions.append(prec)
            test_recalls.append(recall)
            test_f1s.append(f1)


            # Collect the results into a list of dictionaries
            records.append({
                "index": index.item() if isinstance(index, torch.Tensor) else index,
                "accuracy": acc,
                "precision": prec,
                "recall": recall,
                "f1": f1
            })

            # Save X_pred matrix
            X_pred_np = X_pred.detach().cpu().numpy()
            x_pred_filename = os.path.join(x_pred_dir, f'{index[0]}.npy')
            np.save(x_pred_filename, X_pred_np)


    avg_test_loss = test_loss / len(test_loader)

    # STORE METRICS, MODEL CONFIG, AND RESULTS
    # Save the evaluation metrics
    results_df = pd.DataFrame(records)
    results_df.to_csv(f'{output_dir}/pygm_neural/{model}_evaluation_metrics.csv', index=False)
    
    return net, avg_test_loss, test_accuracies, test_precisions, test_recalls, test_f1s
