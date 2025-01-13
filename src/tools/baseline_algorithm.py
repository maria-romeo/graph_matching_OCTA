import numpy as np
from sklearn.neighbors import NearestNeighbors


# Function to calculate Euclidean distance
def euclidean_distance(list_a, list_b):
    return np.sqrt(sum((np.array(list_a) - np.array(list_b)) ** 2))

# Function to normalize an array
def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return [(x - min_val) / (max_val - min_val) for x in array]

# Function to calculate the nearest neighbors
def auto_nearest_neighbor_set(query_points, data_points, n_neighbors):
    nn = NearestNeighbors(algorithm='auto', n_neighbors=n_neighbors)
    nn.fit(data_points)
    distances, indices = nn.kneighbors(query_points, n_neighbors=n_neighbors)
    return indices, distances

def get_features_bif(nodes_bifurc_1, nodes_bifurc_2, points_graph1, points_graph2, valid_feats, feats_dist_list):
    for i in range(len(nodes_bifurc_1)):
        feats_bif_1 = points_graph1[nodes_bifurc_1[i]]['features']
        feats_bif_2 = points_graph2[nodes_bifurc_2[i]]['features']
        key_feats_1 = []
        key_feats_2 = []
        for key in feats_bif_1.keys():
            if key in valid_feats:
                key_feats_1.append(feats_bif_1[key])
                key_feats_2.append(feats_bif_2[key])
        feats_dist_list.append(euclidean_distance(key_feats_1, key_feats_2))
    return feats_dist_list


def spatial_vessel_matching(data_edges, json_edges, valid_feats, threshold_final_dist, n_neighbors, factors_dist):

    data_edges_graph1 = data_edges['graph1']
    data_edges_graph2 = data_edges['graph2']
    json_edges_1 = json_edges['graph1']
    json_edges_2 = json_edges['graph2']

    ## GET THE COORDINATES FOR EACH VESSEL
    points_graph1 = {}
    coords_graph1 = []

    for index, row in data_edges_graph1.iterrows():
        data_point = {}

        bifurcation_point_1 = (row['pos_x_x'], row['pos_y_x'])
        bifurcation_point_2 = (row['pos_x_y'], row['pos_y_y'])
        middle_point = ((bifurcation_point_2[0] + bifurcation_point_1[0]) / 2, (bifurcation_point_2[1] + bifurcation_point_1[1]) / 2)
        coords_graph1.append(middle_point)

        data_point['middle_point'] = middle_point

        data_point['bifurcation_pts'] = [bifurcation_point_1, bifurcation_point_2]

        data_point['features'] = row
        
        data_point['centerline'] = json_edges_1.iloc[index]['pos']

        points_graph1[index] = data_point

    # Create a dictionary for bifurcation points
    bif_pts_graph1 = {}
    # Iterate through each node and its bifurcation points
    for node, data in points_graph1.items():
        for coords in data['bifurcation_pts']:
            if coords not in bif_pts_graph1:
                bif_pts_graph1[coords] = {'nodes': []}
            bif_pts_graph1[coords]['nodes'].append(node)


    points_graph2 = {}
    coords_graph2 = []

    for index, row in data_edges_graph2.iterrows():
        data_point = {}

        bifurcation_point_1 = (row['pos_x_x'], row['pos_y_x'])
        bifurcation_point_2 = (row['pos_x_y'], row['pos_y_y'])
        middle_point = ((bifurcation_point_2[0] + bifurcation_point_1[0]) / 2, (bifurcation_point_2[1] + bifurcation_point_1[1]) / 2)
        coords_graph2.append(middle_point)

        data_point['middle_point'] = middle_point

        data_point['bifurcation_pts'] = [bifurcation_point_1, bifurcation_point_2]

        data_point['features'] = row

        data_point['centerline'] = json_edges_2.iloc[index]['pos']

        points_graph2[index] = data_point

    # Create a dictionary for bifurcation points
    bif_pts_graph2 = {}

    # Iterate through each node and its bifurcation points
    for node, data in points_graph2.items():
        for coords in data['bifurcation_pts']:
            if coords not in bif_pts_graph2:
                bif_pts_graph2[coords] = {'nodes': []}
            bif_pts_graph2[coords]['nodes'].append(node)

    #### COORDINATES
    coords_graph1 = np.array(coords_graph1)
    coords_graph2 = np.array(coords_graph2)

    ## GET THE NEAREST NEIGHBORS FOR EACH POINT IN GRAPH 1
    nearest_neighbors_auto, distances_auto = auto_nearest_neighbor_set(coords_graph1, coords_graph2, n_neighbors)

    distances_auto = distances_auto
    indices_auto = nearest_neighbors_auto

    ## GRAPH MATCHING
    final_idxs_graphs = []
    final_info_graphs = []
    final_coords_graphs = []

    final_matrix = np.zeros((len(coords_graph1), len(coords_graph2)))
    final_matrix[:,:] = -10e6

    for point in range(len(indices_auto)):
        coords_1 = coords_graph1[point]
        bifurc_pts_1 = points_graph1[point]['bifurcation_pts']
        feats_1 = points_graph1[point]['features']
        nodes_bifurc1_1 = bif_pts_graph1[bifurc_pts_1[0]]['nodes']
        nodes_bifurc1_2 = bif_pts_graph1[bifurc_pts_1[1]]['nodes']

        all_distances = []

        final_point_2 = None
        final_dist = threshold_final_dist

        for possible_point in range(n_neighbors):
            point2 = indices_auto[point][possible_point]
            coords_2 = coords_graph2[point2]
            bifurc_pts_2 = points_graph2[point2]['bifurcation_pts']
            feats_2 = points_graph2[point2]['features']

            nodes_bifurc2_1 = bif_pts_graph2[bifurc_pts_2[0]]['nodes']
            nodes_bifurc2_2 = bif_pts_graph2[bifurc_pts_2[1]]['nodes']

            # Get the euclidean distance between the features of the main vessel under study 
            main_vessel_dist_1 = []
            main_vessel_dist_2 = []
            for key in feats_2.keys():
                if key in valid_feats: 
                    main_vessel_dist_1.append(feats_1[key])
                    main_vessel_dist_2.append(feats_2[key])
            main_vessel_dist = euclidean_distance(main_vessel_dist_1, main_vessel_dist_2)

            feats_dist_list = []

            if len(nodes_bifurc1_1) == len(nodes_bifurc2_1) and len(nodes_bifurc1_2) == len(nodes_bifurc2_2): # same number of bifurcation points
                # Get the euclidean distance between the features of the nodes in which the bifurcation points are present
                feats_dist_list = get_features_bif(nodes_bifurc1_1, nodes_bifurc2_1, points_graph1, points_graph2, valid_feats, feats_dist_list)
                feats_dist_list = get_features_bif(nodes_bifurc1_2, nodes_bifurc2_2, points_graph1, points_graph2, valid_feats, feats_dist_list)
                # Get the euclidean distance between the features of the two main nodes, and multiply it by two (more important)
                key_feats_1 = []
                key_feats_2 = []
                for key in feats_2.keys():
                    if key in valid_feats: 
                        key_feats_1.append(feats_1[key])
                        key_feats_2.append(feats_2[key])
                feats_dist_list.append(main_vessel_dist*2) # multiply by 2 to give more importance to the features of the main edge
                all_distances.append(sum(feats_dist_list)/len(feats_dist_list))
            
            elif (len(nodes_bifurc1_1) == len(nodes_bifurc2_1) and len(nodes_bifurc1_2) <= len(nodes_bifurc2_2)) or (len(nodes_bifurc1_1) <= len(nodes_bifurc2_1) and len(nodes_bifurc1_2) == len(nodes_bifurc2_2)): 
                all_distances.append(main_vessel_dist*factors_dist[0])

            elif (len(nodes_bifurc1_1) == len(nodes_bifurc2_1) and len(nodes_bifurc1_2) >= len(nodes_bifurc2_2)) or (len(nodes_bifurc1_1) >= len(nodes_bifurc2_1) and len(nodes_bifurc1_2) == len(nodes_bifurc2_2)): 
                all_distances.append(main_vessel_dist*factors_dist[1])
            
            elif len(nodes_bifurc1_1) <= len(nodes_bifurc2_1) and len(nodes_bifurc1_2) <= len(nodes_bifurc2_2): 
                all_distances.append(main_vessel_dist*factors_dist[2])

            elif len(nodes_bifurc1_1) >= len(nodes_bifurc2_1) and len(nodes_bifurc1_2) >= len(nodes_bifurc2_2):
                all_distances.append(main_vessel_dist*factors_dist[3])
            
            else:
                all_distances.append(main_vessel_dist*factors_dist[4])
        
        min_index = np.argmin(all_distances)
        min_dist = all_distances[min_index]

        if final_dist > min_dist:
            final_point_2 = indices_auto[point][min_index]

        if final_point_2 is not None:
            final_graph2 = points_graph2[final_point_2]
            final_graph1 = points_graph1[point]
            final_idxs = {'graph1': point, 'graph2': final_point_2}
            final_info = {'graph1': final_graph1, 'graph2': final_graph2}
            final_coords = {'graph1': coords_graph1[point], 'graph2': coords_graph2[final_point_2]}

            final_idxs_graphs.append(final_idxs)   
            final_info_graphs.append(final_info)
            final_coords_graphs.append(final_coords)


        final_matrix[point, indices_auto[point]] = 1 - np.array(all_distances)

    
    return final_matrix, final_coords_graphs, final_info_graphs, final_idxs_graphs
