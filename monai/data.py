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
            x,y,z = nib.aff2axcodes(nifti_image.affine)

            # Apply the reorientation
            if x == 'S' or x == 'I':
                axis = 0
            elif y == 'S' or y == 'I':
                axis = 1
            elif z == 'S' or z == 'I':
                axis = 2
            else:
                raise ValueError("Unknown axis orientation")
            
            canonical_affine = nib.as_closest_canonical(nifti_image).affine

            # Extract 2D slices
            depth = nifti_image.shape[axis]
            image_data = nifti_image.get_fdata()
            mask_data = nifti_mask.get_fdata()
            for j in tqdm(range(depth), desc="Loading slices"):
                # Extract the 2D slice and its corresponding mask
                slice = np.take(image_data, j, axis)
                mask = np.take(mask_data, j, axis)
                
                if not (slice.sum() == 0 or mask.sum() == 0):
                    
                    # Correct the header to match the new shape
                    slice_header = nifti_image.header.copy()
                    mask_header = nifti_mask.header.copy()
                    slice_header.set_data_shape(slice.shape)
                    mask_header.set_data_shape(mask.shape)

                    # Add channel dimension
                    slice = slice[np.newaxis, ...]
                    mask = mask[np.newaxis, ...]

                    # Create new NIfTI images (might take too much time and memory, consider other methods.... > TODO)
                    slice = nib.Nifti1Image(slice, canonical_affine, slice_header)
                    mask = nib.Nifti1Image(mask, canonical_affine, mask_header)

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

class Segmentation2dDatasetMulticlass(Dataset):
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
        self.masks_wm = []
        self.masks_gm = []
        self.transform = transform

        n_volumes = len(data)
        for i in range (n_volumes):
            logger.info(f"Loading volume {i+1}/{n_volumes}...")

            # Load the 3D volume
            nifti_image = nib.load(data[i]['image'])
            nifti_mask_wm = nib.load(data[i]['mask_wm'])
            nifti_mask_gm = nib.load(data[i]['mask_gm'])

            # Check if the image and mask shapes match
            assert nifti_image.shape == nifti_mask_gm.shape and nifti_image.shape == nifti_mask_wm.shape, "Image and mask shapes do not match."

            # Make sure the image and mask are in the same canonical space
            x,y,z = nib.aff2axcodes(nifti_image.affine)

            # Apply the reorientation
            if x == 'S' or x == 'I':
                axis = 0
            elif y == 'S' or y == 'I':
                axis = 1
            elif z == 'S' or z == 'I':
                axis = 2
            else:
                raise ValueError("Unknown axis orientation")
            
            canonical_affine = nib.as_closest_canonical(nifti_image).affine

            # Extract 2D slices
            depth = nifti_image.shape[axis]
            image_data = nifti_image.get_fdata()
            mask_data_wm = nifti_mask_wm.get_fdata()
            mask_data_gm = nifti_mask_gm.get_fdata()
            for j in tqdm(range(depth), desc="Loading slices"):
                # Extract the 2D slice and its corresponding mask
                slice = np.take(image_data, j, axis)
                mask_wm = np.take(mask_data_wm, j, axis)
                mask_gm = np.take(mask_data_gm, j, axis)
                
                if not (slice.sum() == 0 or mask_wm.sum() == 0 or mask_gm.sum() == 0):
                    
                    # Correct the header to match the new shape
                    slice_header = nifti_image.header.copy()
                    mask_wm_header = nifti_mask_wm.header.copy()
                    mask_gm_header = nifti_mask_gm.header.copy()
                    slice_header.set_data_shape(slice.shape)
                    mask_wm_header.set_data_shape(mask_wm.shape)
                    mask_gm_header.set_data_shape(mask_gm.shape)


                    # Create new NIfTI images (might take too much time and memory, consider other methods.... > TODO)
                    slice = nib.Nifti1Image(slice, canonical_affine, slice_header)
                    mask_wm = nib.Nifti1Image(mask_wm, canonical_affine, mask_wm_header)
                    mask_gm = nib.Nifti1Image(mask_gm, canonical_affine, mask_gm_header)

                    # Append the 2D slice and its corresponding mask to the dataset, as Nifti1Images
                    self.slices.append(slice)
                    self.masks_wm.append(mask_wm)
                    self.masks_gm.append(mask_gm)  
                
        logger.info("Dataset loaded!")          

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the 2D slice to be returned.

        Returns:
            dict: A dictionnary containing the 2D slice and its corresponding mask.
        """
        slice = self.slices[index]
        mask_wm = self.masks_wm[index]
        mask_gm = self.masks_gm[index]

        sample = {"image": slice, "mask_wm": mask_wm, "mask_gm": mask_gm}

        # Apply transformations if any
        if self.transform:
           sample = self.transform(sample)

        return sample

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.slices)


def remove_empty_slices(images, masks):
    valid_indices = [i for i, (image, mask) in enumerate(zip(images, masks)) if not (image.sum() == 0 or mask.sum() == 0)]
    filtered_images = [images[i] for i in valid_indices]
    filtered_masks = [masks[i] for i in valid_indices]
    return filtered_images, filtered_masks