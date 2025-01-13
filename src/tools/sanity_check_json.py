import pandas as pd
import json 

# SANITY CHECK FOR JSON FILES

def sanity_check_json(nodes_file, edges_file, json_file, new_json_file):
    # Load the CSV files
    df_nodes = pd.read_csv(nodes_file, delimiter=';')  # Update with the correct file path for nodes CSV
    df_edges = pd.read_csv(edges_file, delimiter=';')  # Update with the correct file path for edges CSV

    # Load the JSON file
    with open(json_file, 'r') as f:  # Update with the correct file path for JSON file
        graph_data = json.load(f)

    """# Filter nodes in JSON
    filtered_nodes = []
    new_node_id = 0

    for node in graph_data['graph']['nodes']:
        if node['pos'][0] == df_nodes['pos_x'][new_node_id] and node['pos'][1] == df_nodes['pos_y'][new_node_id]:
            node['id'] = new_node_id
            filtered_nodes.append(node)
            new_node_id += 1
        if new_node_id == len(df_nodes):
            break"""

    # Filter edges in JSON
    filtered_edges = []
    new_edge_id = 0

    for edge in graph_data['graph']['edges']:
        node1 = edge['node1']
        node2 = edge['node2']
        # Check if both nodes exist in the filtered list and if the pair exists in CSV edges
        if (node1, node2) == (df_edges['node1id'][new_edge_id], df_edges['node2id'][new_edge_id]):
            edge['id'] = new_edge_id
            filtered_edges.append(edge)
            new_edge_id += 1
        if new_edge_id == len(df_edges):
            break
            

    # Update graph data
    #graph_data['graph']['nodes'] = filtered_nodes
    graph_data['graph']['edges'] = filtered_edges

    # Save the updated JSON
    with open(new_json_file, 'w') as f:  # Update with the correct output file path
        json.dump(graph_data, f, indent=4)

    print("Filtered JSON file has been saved.")
