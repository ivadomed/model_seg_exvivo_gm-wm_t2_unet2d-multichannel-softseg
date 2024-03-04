import nibabel as nib
import numpy as np

from monai.data import Dataset
from transforms import remove_empty_slices
from skimage.transform import resize

class Segmentation2dDataset(Dataset):
    """
    A custom Dataset class that loads 3D NIfTI volumes and returns 2D slices.
    """
    def __init__(self, data, axis, transform=None):
        """
        Args:
            data (list): List of dictionnaries containing the file paths of the 3D volumes.
            axis (str): Axis to extract the 2D slices from.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.slices = []
        self.masks = []
        self.transform = transform

        # Load the 3D volumes and their corresponding masks
        for i in range(len(data)):
            image, mask = load_2d_training_sample(data[i]['image'], data[i]['mask'], axis)

            image = np.moveaxis(image, axis, 0)
            mask = np.moveaxis(mask, axis, 0)
            for j in range(image.shape[0]):
                self.slices.append(image[j,:, :])
                self.masks.append(mask[j, :,:])
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the 2D slice to be returned.

        Returns:
            dict: A dictionnary containing the 2D slice and its corresponding mask.
        """
        slice = self.slices[index]
        mask = self.masks[index]

        # Resampling
        slice = resize(slice, (200,200), anti_aliasing=True)
        mask = resize(mask, (200,200), anti_aliasing=True)

        # Apply transformations if any
        if self.transform:
            slice = self.transform(slice)
            mask = self.transform(mask)

        return {"image": slice, "mask": mask}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.slices)

class Inference2dDataset(Dataset):
    """
    A custom Dataset class that loads 3D NIfTI volumes and returns 2D slices.
    """
    def __init__(self, data, axis, transform=None):
        """
        Args:
            data (list): List of dictionnaries containing the file paths of the 3D volumes.
            axis (str): Axis to extract the 2D slices from.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.slices = []
        self.transform = transform

        # Load the 3D volumes 
        for i in range(len(data)):
            image = nib.load(data[i]['image'])
            image = image.get_fdata()
            image = np.moveaxis(image, axis, 0)
            for j in range(image.shape[0]):
                self.slices.append(image[j])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the 2D slice to be returned.

        Returns:
            dict: A dictionnary containing the 2D slice and its corresponding mask.
        """
        slice = self.slices[index]

        # Resampling
        slice = resize(slice, (200,200), anti_aliasing=True)

        # Apply transformations if any
        if self.transform:
            slice = self.transform(slice)

        return {"image": slice}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.slices)


def load_2d_training_sample(image_path, mask_path, axis):   
    # Load the 3D volume
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    image_data = image.get_fdata()
    mask_data = mask.get_fdata()

    # Remove empty slices
    image_data, mask_data = np.moveaxis(image_data, axis, 0), np.moveaxis(mask_data, axis, 0)
    image_data, mask_data = remove_empty_slices(image_data, mask_data)
    image_data, mask_data = np.moveaxis(image_data, 0, axis), np.moveaxis(mask_data, 0, axis)

    return image_data, mask_data

    
