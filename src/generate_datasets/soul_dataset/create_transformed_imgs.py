import numpy as np
import os
import random 
import pandas as pd
from PIL import Image

import sys
sys.path.append(os.path.abspath('../../'))
from transformation_functions import *


# List of transformations
transformations = ['translate', 'rotate', 'elastic'] 

translation_range = [-15, 15]
rotation_range = [-10, 10]
alpha_range = [5, 40]

src_dir = '/Users/maria/Documents/GitHub/TFM/OCTA_time_series/GRAPH_matching/data/soul_dataset_big/'

segs_dir = os.path.join(src_dir, 'segmentations')

# Initialize lists to store transformations
idxs = []
transform_types = []
txs = []
tys = []
angles = []
alphas = []

for seg in os.listdir(segs_dir):
    if seg == '.DS_Store':
        continue

    # Load segmentation
    segmentation = np.array(Image.open(os.path.join(segs_dir, seg)))

    # Generate unique random integers within the specified ranges
    trans_params = random.sample(range(translation_range[0], translation_range[1] + 1), 20) 
    rot_params = random.sample(range(rotation_range[0], rotation_range[1] + 1), 20) 
    alpha_params = random.sample(range(alpha_range[0], alpha_range[1] + 1), 20) 

    # Apply transformations
    for T in transformations:
        for i in range(20):
            if T == 'translate':
                tx, ty = trans_params[i], trans_params[-i-1]
                angle, alpha = None, None
                idx = i 
                trans = translate(segmentation, tx, ty)
            elif T == 'rotate':
                angle = rot_params[i]
                tx, ty, alpha = None, None, None
                trans = rotate(segmentation, segmentation.shape[0], segmentation.shape[1], angle)
                idx = i+20
            elif T == 'elastic':
                alpha = alpha_params[i]
                tx, ty, angle = None, None, None
                idx = i+40
                trans = elastic(segmentation, alpha, os.path.join(src_dir, 'elastic_def_matrix'), seg.split('.')[0] + '_' + str(idx))
            
            Image.fromarray(trans, mode='L').save(os.path.join(src_dir, 'transformed_images', seg.split('.')[0] + '_' + str(idx) + '.png'))

            idxs.append(seg.split('.')[0] + '_' + str(idx))
            transform_types.append(T)
            txs.append(tx)
            tys.append(ty)
            angles.append(angle)
            alphas.append(alpha)
        
df_transformations = pd.DataFrame({'index': idxs, 'transform_type': transform_types, 'tx':txs, 'ty':tys, 'angle':angles, 'alpha': alphas})
# Save transformations log to a CSV file
log_csv_path = os.path.join(src_dir, "transformations_log.csv")
df_transformations.to_csv(log_csv_path, index=False)