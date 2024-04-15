import numpy as np
import torch
import nibabel as nib

from monai.transforms import Compose, RandAffined, Rand2DElasticd, Spacingd, NormalizeIntensityd, InvertibleTransform, RandScaleIntensityd, RandGaussianNoised
from monai.data import MetaTensor

def training_transforms(pixdim=(0.1,0.1), translate_range=(-0.1, 0.1), rotate_range=45):
    transforms_list = [
        ToMetaTensord(keys=["image", "mask"]),
        Spacingd(keys=["image", "mask"], pixdim=pixdim, align_corners=True, mode=("bilinear", "nearest"), dtype=np.float32, scale_extent=True), 
        #Data augmentation
        RandAffined(keys=["image", "mask"], 
                    prob=0.9, 
                    translate_range= translate_range, 
                    rotate_range= (-rotate_range / 360 * 2. * np.pi, rotate_range / 360 * 2. * np.pi), 
                    padding_mode="zeros"),
        Rand2DElasticd(keys=["image", "mask"], prob=0.5, spacing=(15, 15), magnitude_range=(1, 2)),
        RandGaussianNoised(keys=["image", "mask"], prob=0.4, mean=0.0, std=0.03),
        #RandGaussianSmoothd(keys=["image", "mask"], prob=0.5, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0)),
        RandScaleIntensityd(keys=["image"], prob=0.5, factors=(0.5, 1.5)),

        NormalizeIntensityd(keys=["image"]),
    ]
    return Compose(transforms_list)

def training_transforms_multiclass(pixdim=(0.1,0.1), translate_range=(-0.1, 0.1), rotate_range=45):
    transforms_list = [
        ToMetaTensord(keys=["image", "mask_wm", "mask_gm"]),
        Spacingd(keys=["image", "mask_wm", "mask_gm"], pixdim=pixdim, align_corners=True, mode=("bilinear", "nearest", "nearest"), dtype=np.float32, scale_extent=True), 
        #Data augmentation
        RandAffined(keys=["image", "mask_wm", "mask_gm"], 
                    prob=0.9, 
                    translate_range= translate_range, 
                    rotate_range= (-rotate_range / 360 * 2. * np.pi, rotate_range / 360 * 2. * np.pi), 
                    padding_mode="zeros"),
        Rand2DElasticd(keys=["image", "mask_wm", "mask_gm"], prob=0.5, spacing=(15, 15), magnitude_range=(1, 2)),
        RandGaussianNoised(keys=["image", "mask_wm", "mask_gm"], prob=0.4, mean=0.0, std=0.03),
        #RandGaussianSmoothd(keys=["image", "mask"], prob=0.5, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0)),
        RandScaleIntensityd(keys=["image"], prob=0.5, factors=(0.5, 1.5)),

        NormalizeIntensityd(keys=["image"]),
    ]
    return Compose(transforms_list)

class ToMetaTensord(InvertibleTransform):
    """
    Transform to convert NIfTI files to MetaTensors, keeping the metadata.
    """
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            nifti_img = (data[key])

            tensor_data = torch.from_numpy(nifti_img.get_fdata().astype('float32'))

            #add channel dimension if missing
            if len(tensor_data.shape) == 2:
                tensor_data = tensor_data.unsqueeze(0)

            meta_tensor = MetaTensor(tensor_data, affine =  nifti_img.affine)

            data[key] = meta_tensor
        return data
    
    def inverse(self, data):
        for key in self.keys:
            meta_tensor = data[key]
            tensor_data = meta_tensor.data
            nifti_img = nib.Nifti1Image(tensor_data.cpu().numpy(), meta_tensor.affine)
            data[key] = nifti_img
        return data
    
