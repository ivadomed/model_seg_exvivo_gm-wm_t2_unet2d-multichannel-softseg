import nibabel as nib
import numpy as np

from loguru import logger
from tqdm import tqdm

from monai.data import Dataset

class Segmentation2dDataset(Dataset):
    """
    A custom Dataset class that loads 3D NIfTI volumes and returns 2D slices.
    """
    def __init__(self, data, transform=None):
        """
        Args:
            data (list): List of dictionnaries containing the file paths of the 3D volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.slices = []
        self.masks = []
        self.transform = transform

        n_volumes = len(data)
        for i in range (n_volumes):
            logger.info(f"Loading volume {i+1}/{n_volumes}...")

            # Load the 3D volume
            nifti_image = nib.load(data[i]['image'])
            nifti_mask = nib.load(data[i]['mask'])

            # Check if the image and mask shapes match
            assert nifti_image.shape == nifti_mask.shape, "Image and mask shapes do not match."

            # Make sure the image and mask are in the same canonical space
            nifti_image = nib.as_closest_canonical(nifti_image)
            nifti_mask = nib.as_closest_canonical(nifti_mask)

            # Extract 2D slices
            depth = nifti_image.shape[2]
            image_data = nifti_image.get_fdata()
            mask_data = nifti_mask.get_fdata()
            for j in tqdm(range(depth), desc="Loading slices"):
                # Extract the 2D slice and its corresponding mask
                slice = np.take(image_data, j, 2)
                mask = np.take(mask_data, j, 2)
                
                if not (slice.sum() == 0 or mask.sum() == 0):
                    # Add channel dimension
                    slice = slice[np.newaxis, ...]
                    mask = mask[np.newaxis, ...]

                    # Correct the header to match the new shape
                    slice_header = nifti_image.header.copy()
                    mask_header = nifti_mask.header.copy()
                    slice_header.set_data_shape(slice.shape)
                    mask_header.set_data_shape(mask.shape)

                    # Create new NIfTI images (might take too much time and memory, consider other methods.... > TODO)
                    slice = nib.Nifti1Image(slice, nifti_image.affine, slice_header)
                    mask = nib.Nifti1Image(mask, nifti_mask.affine, mask_header)

                    # Append the 2D slice and its corresponding mask to the dataset, as Nifti1Images
                    self.slices.append(slice)
                    self.masks.append(mask)  
                
        logger.info("Dataset loaded!")          

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the 2D slice to be returned.

        Returns:
            dict: A dictionnary containing the 2D slice and its corresponding mask.
        """
        slice = self.slices[index]
        mask = self.masks[index]

        sample = {"image": slice, "mask": mask}

        # Apply transformations if any
        if self.transform:
           sample = self.transform(sample)

        return sample

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.slices)

class Inference2dDataset(Dataset):
    """
    A custom Dataset class that loads 3D NIfTI volumes and returns 2D slices.
    """
    def __init__(self, data, transform=None):
        """
        Args:
            data (list): List of dictionnaries containing the file paths of the 3D volumes.
            axis (str): Axis to extract the 2D slices from.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.slices = []
        self.transform = transform
        
        logger.info(f"Loading volume...")

        # Load the 3D volume
        nifti_image = nib.load(data['image'])

        # Make sure the image is in the right canonical space
        nifti_image = nib.as_closest_canonical(nifti_image)
        # Extract 2D slices
        depth = nifti_image.shape[2]
        image_data = nifti_image.get_fdata()
        for j in tqdm(range(depth), desc="Loading slices"):
            # Extract the 2D slice 
            slice = np.take(image_data, j, 2)
            
            # Add channel dimension
            slice = slice[np.newaxis, ...]

            # Correct the header to match the new shape
            slice_header = nifti_image.header.copy()
            slice_header.set_data_shape(slice.shape)

            # Create new NIfTI images (might take too much time and memory, consider other methods.... > TODO)
            slice = nib.Nifti1Image(slice, nifti_image.affine, slice_header)

            # Append the 2D slice and its corresponding mask to the dataset, as Nifti1Images
            self.slices.append(slice)
            
        logger.info("Image loaded!")          


    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the 2D slice to be returned.

        Returns:
            dict: A dictionnary containing the 2D slice.
        """
        slice = self.slices[index]
        sample = {"image": slice}

        # Apply transformations if any
        if self.transform:
            sample = self.transform(sample)

        return sample

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

    
def remove_empty_slices(images, masks):
    valid_indices = [i for i, (image, mask) in enumerate(zip(images, masks)) if not (image.sum() == 0 or mask.sum() == 0)]
    filtered_images = [images[i] for i in valid_indices]
    filtered_masks = [masks[i] for i in valid_indices]
    return filtered_images, filtered_masks