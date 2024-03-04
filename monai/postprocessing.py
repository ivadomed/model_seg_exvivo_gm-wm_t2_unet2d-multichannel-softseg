from scipy.ndimage import label, generate_binary_structure
import numpy as np

def remove_noise(seg_map, thr):
    """Remove prediction values under the given threshold

    Args:
        seg_map (array): Segmentation map.
        thr (float): Threshold under which predictions are set to 0.

    """
    if thr >= 0:
        mask = seg_map > thr
        return seg_map * mask
    else:
        return seg_map
    
def remove_small(seg_map, thr):
    """Remove small objects

    Args:
        seg_map (array): Segmentation map.  
        thr (int or list): Minimal object size to keep in input data.

    """
    bin_structure = generate_binary_structure(2, 1)
    data_label, n = label(seg_map, structure=bin_structure)

    for idx in range(1, n + 1):
        data_idx = (data_label == idx).astype(int)
        n_nonzero = np.count_nonzero(data_idx)

        if n_nonzero < thr:
            seg_map[data_label == idx] = 0

    return seg_map
    