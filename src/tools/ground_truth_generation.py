import numpy as np
from skimage import morphology

def labeled_centerline_array(base_img, graph_points, labels):
    labeled_centerline = np.zeros_like(base_img)
    dict_labels = {}

    for vessel in graph_points:
        vessel_coords = np.array(graph_points[vessel]['centerline'])
        vessel_coords = np.round(vessel_coords).astype(int)
        labeled_centerline[vessel_coords[:, 0], vessel_coords[:, 1]] = labels[vessel]
        dict_labels[labels[vessel]] = vessel

    return labeled_centerline, dict_labels

def labeled_image(seg, cl):

    seg[seg != 0] = 1
    final_seg_label = np.zeros_like(seg, dtype=np.uint16)
    final_seg_label[seg != 0] = 1

    cl_vessel = cl.copy()

    label_cl = cl_vessel  # measure.label(cl_vessel)
    label_cl[label_cl != 0] = label_cl[label_cl != 0] + 1 # it was + 1
    final_seg_label[label_cl != 0] = label_cl[label_cl != 0]

    for i in range(100):
        label_cl = morphology.dilation(label_cl, morphology.square(3))

        label_cl = label_cl * seg

        # get the values of final_seg_label where no semantic segmentation is present
        final_seg_label[final_seg_label == 1] = label_cl[final_seg_label == 1]
        # get indices where label_cl==0 and seg !=0
        mask = (final_seg_label == 0) & (seg != 0)
        final_seg_label[mask] = 1

    # pixels that are still 1 are turned into 0
    final_seg_label[final_seg_label == 1] = 0
    # labels for the rest are corrected by -1
    final_seg_label[final_seg_label != 0] = (
        final_seg_label[final_seg_label != 0] - 1
    )

    return final_seg_label

# Create a matching dictionary for the ground truth
def create_matching(labels, graph1_label_image, inverse_graph2_label_image, labels1_to_graph1, labels2_to_graph2):
    matching_dict = {}
    for label in labels:
        if label == 0:
            continue
        coords_label1 = np.where(graph1_label_image == label)
        
        vals_g2 = inverse_graph2_label_image[coords_label1]
        
        unique_vals,  counts = np.unique(vals_g2, return_counts=True)

        if len(counts) != 0:
            # Get the most common value
            max_count = np.max(counts)
            max_index = np.argmax(counts)
            area = max_count/len(coords_label1[0])

            if area > 0.5:
                label2 = unique_vals[max_index]
            else:
                label2 = None
        
        else:
            label2 = None

        if label2 != 0 and label2 != None:
            vessel2_id = labels2_to_graph2[label2]
            vessel1_id = labels1_to_graph1[label]
            matching_dict[vessel1_id] = vessel2_id

    return matching_dict

def create_gt(points_graph1, color_graph1, seg1, points_graph2, color_graph2, seg2, inverse_transform, transform_type, parameters):
    np.random.seed(42)
    # Ensure unique labels for each vessel
    labels_1 = np.arange(100, 100 + len(points_graph1))
    np.random.shuffle(labels_1)
    graph1_label_cl, labels1_to_graph1 = labeled_centerline_array(color_graph1, points_graph1, labels_1)

    labels_2 = np.arange(100, 100 + len(points_graph2))
    np.random.shuffle(labels_2)
    graph2_label_cl, labels2_to_graph2 = labeled_centerline_array(color_graph2, points_graph2, labels_2)

    graph1_label_img = labeled_image(seg1, graph1_label_cl)
    graph2_label_img = labeled_image(seg2, graph2_label_cl)

    inverse_graph2_label_img = inverse_transform(transform_type, graph2_label_img, parameters)

    matching_1to2 = create_matching(labels_1, graph1_label_img, inverse_graph2_label_img, labels1_to_graph1, labels2_to_graph2)
    matching_2to1 = create_matching(labels_2, inverse_graph2_label_img, graph1_label_img, labels2_to_graph2, labels1_to_graph1)

    # Compare both dictionaries and select as ground truth only the matches that are equal in both matchings
    gt_1to2 = {}
    for vessel1, vessel2 in matching_1to2.items():
        # Get the vessel 1 for that vessel 2 in the matching_2to1 dict
        vessel1_2to1 = matching_2to1.get(vessel2)
        if vessel1_2to1 == vessel1:
            gt_1to2[vessel1] = vessel2

    return gt_1to2
                