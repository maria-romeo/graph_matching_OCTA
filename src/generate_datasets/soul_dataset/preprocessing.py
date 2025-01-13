import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from PIL import Image
import os 

dir = '/Users/maria/Documents/GitHub/TFM/OCTA_time_series/GRAPH_matching/data/soul_dataset_big/soul_imgs_jpg/' 
output_dir = '/Users/maria/Documents/GitHub/TFM/OCTA_time_series/GRAPH_matching/data/soul_dataset_big/segmentations/'

for img in os.listdir(dir):
    if img == '.DS_Store':
        continue
    # Load the image
    seg_dir = dir + img
    seg = np.array(Image.open(seg_dir).convert('L'))

    binary_image = np.where(seg > 10, 255, 0).astype(np.uint8)

    # Convert the image to binary
    binary_segmentation = binary_image == 255  # Creates a boolean array where 255 becomes True

    # Invert the binary image (vessels become 0, background becomes 1)
    #inverted_segmentation = np.logical_not(binary_segmentation)

    # Apply binary_fill_holes to fill only small holes inside the vessel structure
    filled_inverted = binary_dilation(binary_segmentation, iterations=1)
    filled_inverted = binary_erosion(filled_inverted, iterations=1)
    filled_inverted = binary_dilation(filled_inverted, iterations=1)
    filled_inverted = binary_erosion(filled_inverted, iterations=1)

    # Invert the result back
    #filled_segmentation = np.logical_not(filled_inverted)
    filled_segmentation = filled_inverted

    # Convert back to uint8 format
    filled_segmentation = filled_segmentation.astype(np.uint8)*255

    # Save the filled segmentation
    new_name = img.split('_')[0] + '.png'
    Image.fromarray(filled_segmentation, mode='L').save(output_dir + new_name)