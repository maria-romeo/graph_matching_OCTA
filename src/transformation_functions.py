
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import os


def translate(img, tx, ty):
    # Shift the array
    transformed_array = np.roll(img, tx, axis=1)
    transformed_array = np.roll(transformed_array, ty, axis=0)
    return transformed_array

def recover_translation(img, tx, ty):
    # Shift the array back by negative offsets
    recovered_array = np.roll(img, -tx, axis=1)
    recovered_array = np.roll(recovered_array, -ty, axis=0)
    return recovered_array

def rotate(img, height, width, angle):
    # Create an empty array for the transformed image
    transformed_array = np.zeros_like(img)
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2
    # Create coordinate arrays
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    # Calculate coordinates relative to the center
    x_rel, y_rel = x - center_x, y - center_y
    # Apply rotation matrix
    new_x = center_x + (x_rel * np.cos(angle_rad) - y_rel * np.sin(angle_rad))
    new_y = center_y + (x_rel * np.sin(angle_rad) + y_rel * np.cos(angle_rad))
    # Flatten the coordinate arrays for map_coordinates
    coords = np.array([new_y.flatten(), new_x.flatten()])
    # Apply map_coordinates to interpolate the image at the new coordinates
    transformed_array = map_coordinates(img, coords, order=0, mode='constant', cval=0).reshape((height, width))
    return transformed_array

def recover_rotation(img, height, width, angle):
    # Convert angle to radians and negate it for inverse rotation
    angle_rad = np.deg2rad(-angle)
    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2
    # Create coordinate arrays
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    # Calculate coordinates relative to the center
    x_rel, y_rel = x - center_x, y - center_y
    # Apply inverse rotation matrix
    new_x = center_x + (x_rel * np.cos(angle_rad) - y_rel * np.sin(angle_rad))
    new_y = center_y + (x_rel * np.sin(angle_rad) + y_rel * np.cos(angle_rad))
    # Flatten the coordinate arrays for map_coordinates
    coords = np.array([new_y.flatten(), new_x.flatten()])
    # Apply map_coordinates to interpolate the image at the new coordinates
    recovered_array = map_coordinates(img, coords, order=0, mode='constant', cval=0).reshape((height, width))
    return recovered_array


def elastic(img, alpha, output_def, idx):
    random_state = np.random.RandomState(None)
    shape = img.shape[:2]
    sigma = 5 
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    transformed_array = map_coordinates(img, indices, order=0, mode='constant', cval=0).reshape(shape)
    # Store dx and dy
    np.save((os.path.join(output_def, 'dx_'+str(idx)+'.npy')), dx)
    np.save((os.path.join(output_def, 'dy_'+str(idx)+'.npy')), dy)
    return transformed_array

def recover_elastic(img, output_def, idx):
    # Load the displacement fields
    dx = np.load(os.path.join(output_def, f'dx_{idx}.npy'))
    dy = np.load(os.path.join(output_def, f'dy_{idx}.npy'))
    shape = img.shape[:2]
    # Create coordinate arrays
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # Subtract the displacement fields to reverse the deformation
    indices = np.reshape(y - dy, (-1, 1)), np.reshape(x - dx, (-1, 1))
    # Apply map_coordinates to interpolate the image at the new coordinates
    recovered_array = map_coordinates(img, indices, order=0, mode='constant', cval=0).reshape(shape)
    return recovered_array
