def calculate_metrics(final_idxs_graphs, gt_1to2, json_edges_trans):

    FN = 0
    TP = 0
    FP = 0
    TN = 0

    for index, row in json_edges_trans.iterrows():
        idx = row['id']

        exist_gt = False
        exist_res = False

        for gt_key, gt_value in gt_1to2.items():
            if gt_value == idx:
                exist_gt = True
                gt = gt_key
                break
        
        for pred in final_idxs_graphs:
            if pred['graph2'] == idx:
                exist_res = True
                prediction = pred['graph1']
                break
        
        if exist_gt:
            if not exist_res:
                FN += 1 
            else:
                if gt == prediction:
                    TP += 1
                else:
                    FP += 1
        else:
            if exist_res:
                FP += 1
            else:
                TN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return accuracy, precision, recall, f1
