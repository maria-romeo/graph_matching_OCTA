
def get_info_graph(data_edges_graph, json_edges):
    points_graph = {}

    for index, row in data_edges_graph.iterrows():
        data_point = {}

        bifurcation_point_1 = (row['pos_x_x'], row['pos_y_x'])
        bifurcation_point_2 = (row['pos_x_y'], row['pos_y_y'])
        middle_point = ((bifurcation_point_2[0] + bifurcation_point_1[0]) / 2, (bifurcation_point_2[1] + bifurcation_point_1[1]) / 2)

        data_point['middle_point'] = middle_point

        data_point['bifurcation_pts'] = [bifurcation_point_1, bifurcation_point_2]

        data_point['features'] = row
        
        data_point['centerline'] = json_edges.iloc[index]['pos']

        points_graph[index] = data_point

    # Create a dictionary for bifurcation points
    bif_pts_graph = {}
    # Iterate through each node and its bifurcation points
    for node, data in points_graph.items():
        for coords in data['bifurcation_pts']:
            if coords not in bif_pts_graph:
                bif_pts_graph[coords] = {'nodes': []}
            bif_pts_graph[coords]['nodes'].append(node)

    return points_graph, bif_pts_graph