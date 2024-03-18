
import numpy as np

from data import remove_empty_slices

def dice_score(prediction, groundtruth):
    smooth = 1.
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice

def compute_dice_score(prediction, groundtruth): 
    if prediction.shape != groundtruth.shape:
        raise ValueError("Prediction and groundtruth should have the same shape")
    else:
        predictions = []
        groundtruths = []
        for i in range(prediction.shape[1]):
            predictions.append(prediction[:,i,:])
            groundtruths.append(groundtruth[:,i,:])
        
        predictions, groundtruths = remove_empty_slices(predictions, groundtruths)
        predictions, groundtruths = np.array(predictions), np.array(groundtruths)

        return dice_score(predictions, groundtruths)