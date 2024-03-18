import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def check_empty_patch(labels):
    for i, label in enumerate(labels):
        if torch.sum(label) == 0.0:
            # print(f"Empty label patch found at index {i}. Skipping training step ...")
            return None
    return labels  # If no empty patch is found, return the labels

def plot_slices(image, gt, pred, debug=False):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy
    image = image.numpy()
    gt = gt.numpy()
    pred = pred.numpy()

   # if not debug:
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    fig.suptitle('Original Image --> Ground Truth --> Prediction')
    axs[0].imshow(image, cmap='gray'); axs[0].axis('off') 
    axs[1].imshow(gt); axs[1].axis('off')
    axs[2].imshow(pred); axs[2].axis('off')
    

    plt.tight_layout()
    fig.show()
    return fig

# Copied from ivadomed
def keep_largest_object(predictions):
    """Keep the largest connected object from the input array (2D or 3D).

    Args:
        predictions (ndarray or nibabel object): Input segmentation. Image could be 2D or 3D.

    Returns:
        ndarray or nibabel (same object as the input).
    """
    # Find number of closed objects using skimage "label"
    labeled_obj, num_obj = ndimage.label(np.copy(predictions))
    # If more than one object is found, keep the largest one
    if num_obj > 1:
        # Keep the largest object
        predictions[np.where(labeled_obj != (np.bincount(labeled_obj.flat)[1:].argmax() + 1))] = 0
    return predictions

def get_last_folder_id(parent_dir):
    
    if not os.path.exists(parent_dir):
        print(f"The directory {parent_dir} does not exist.")
        return

    # Find the highest numbered directory
    highest_num = 0
    for item in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, item)) and item.isdigit():
            highest_num = max(highest_num, int(item))
            
    return highest_num

def create_indexed_dir(parent_dir):
    highest_num = get_last_folder_id(parent_dir)
    next_dir_num = highest_num + 1
    next_dir_path = os.path.join(parent_dir, str(next_dir_num))
    os.makedirs(next_dir_path, exist_ok=True)
    print(f"Created directory: {next_dir_path}")
    return next_dir_path