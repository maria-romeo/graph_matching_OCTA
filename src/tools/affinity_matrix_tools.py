import torch
import numpy as np
import pygmtools as pygm
import functools

"""def get_connected_pairs_(bif_pts_graph):
    pairs = []
    for bif_pt in bif_pts_graph.values():
        nodes = bif_pt['nodes']
        # Generate all possible directed pairs
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    pairs.append((nodes[i], nodes[j]))
    return torch.from_numpy(np.array(pairs))"""

# Create a list with all the connected pairs
def get_connected_pairs(bif_pts_graph):
    pairs = []
    bif_pairs = {}
    for idx, (coord, bif_pt) in enumerate(bif_pts_graph.items()):
        nodes = bif_pt['nodes']
        # Generate all possible directed pairs
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    pairs.append((nodes[i], nodes[j]))
                    bif_pairs[(nodes[i], nodes[j])] = coord
    return torch.from_numpy(np.array(pairs)), bif_pairs

# Function to extract node features
def extract_node_features(points, features):
    feature_array = []
    for point in points.values():
        feature_values = [point['features'][feat] for feat in features]
        feature_values.append(point['middle_point'][0])
        feature_values.append(point['middle_point'][1]) 
        feature_array.append(feature_values)
    return torch.from_numpy(np.array(feature_array))

# Extract edge features (coordinates of the bifurcation points) 
def extract_edge_features(bif_pts_graph):
    edge_features = []
    for bif_pt, nodes in bif_pts_graph.items():
        edge_features.append([bif_pt[0], bif_pt[1]])#, len(nodes['nodes'])])   
    return torch.from_numpy(np.array(edge_features))

# Extract edge features (coordinates of the bifurcation points) and number of vessels per bifurcation point
def extract_edge_features_matrix(dict_bif_pairs, A):
    edge_matrix = np.zeros((A.shape[0], A.shape[1], 1))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1 and i != j:
                """edge_matrix[i, j, 0] = dict_bif_pairs[(i, j)][0]
                edge_matrix[i, j, 1] = dict_bif_pairs[(i, j)][1]"""
                # Modified bc pygm.cie with pretrained weights only accepts 1 feat
                # combined_feat = (dict_bif_pairs[(i, j)][0] + dict_bif_pairs[(i, j)][1]) / 2 # this wasnt working
                # edge_matrix[i, j, 0] = combined_feat
                edge_matrix[i, j, 0] = 1

    return torch.from_numpy(edge_matrix)

def build_adjacency_matrix(conn, n):
    # Initialize adjacency matrix
    A = np.zeros((n, n))
    # Populate adjacency matrix
    for (i, j) in conn:
        A[i, j] = 1
    return torch.from_numpy(A)

def extract_graph_features(points_graph, bif_pts_graph, valid_feats, n):
    conn, bif_pairs_dict = get_connected_pairs(bif_pts_graph) # connectivity matrix for graph 1
    node_feat = extract_node_features(points_graph, valid_feats)
    edge_feat = extract_edge_features(bif_pts_graph)
    A = build_adjacency_matrix(conn, n)
    edge_feat_matrix = extract_edge_features_matrix(bif_pairs_dict, A)
    return A, node_feat, edge_feat, edge_feat_matrix, conn
