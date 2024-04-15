import os
import numpy as np
from loguru import logger
import torch.nn.functional as F
import torch
import yaml
import nibabel as nib
import shutil
import tempfile
from tqdm import tqdm
from scipy import ndimage

from datetime import datetime
from time import time

from monai.data import (Dataset, DataLoader, decollate_batch)
from monai.transforms import Invertd, Compose, EnsureTyped, LoadImaged, Spacingd, NormalizeIntensityd, EnsureChannelFirstd
from monai.networks.nets import UNet

# ===========================================================================
#                   Prepare temporary dataset for inference
# ===========================================================================
def prepare_data(path_img, pixdim): 
        
        nib_image = nib.load(path_img)
        
        # define inference_transforms
        transform = inference_transforms(pixdim=pixdim) 
        
        logger.info(f"Loading image: {path_img}") 

        tmpdir_slices = tmp_create(basename="sct_deepseg_extract_slices")
        data = extract_2d_data(path_img, tmpdir_slices, key = "image")

        dataset = Dataset(data=data, transform=transform)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        test_post_pred = Compose([
                            EnsureTyped(keys=["pred"]),
                            Invertd(keys=["pred"], transform=transform,
                                    orig_keys=["image"],
                                    meta_keys=["pred_meta_dict"],
                                    nearest_interp=False, to_tensor=True),
                        ])

        return test_loader, test_post_pred, nib_image

def extract_2d_data(path_img, tmp_dir, key):
    logger.info(f"Loading volume...")

    # Load the 3D volume
    nifti_image = nib.load(path_img)

    # Make sure the image is in the right canonical space
    #nifti_image = nib.as_closest_canonical(nifti_image)

    data = []

    # Extract 2D slices
    x,y,z = nib.aff2axcodes(nifti_image.affine)

    # chose the right axis for the slices
    # TODO: use sct function to reorient the image
    if x == 'S' or x == 'I':
        axis = 0
    elif y == 'S' or y == 'I':
        axis = 1
    elif z == 'S' or z == 'I':
        axis = 2
    else:
        raise ValueError("Unknown axis orientation")
    
    canonical_affine = nib.as_closest_canonical(nifti_image).affine

    depth = nifti_image.shape[axis]
    image_data = nifti_image.get_fdata()

    for j in tqdm(range(depth), desc="Loading slices"):
        # Extract the 2D slice 
        slice = np.take(image_data, j, axis)

        # Correct the header to match the new shape
        slice_header = nifti_image.header.copy()
        slice_header.set_data_shape(slice.shape)

        # Create new NIfTI images (might take too much time and memory, consider other methods.... > TODO)
        slice = nib.Nifti1Image(slice, canonical_affine, slice_header)
        path_slice = os.path.join(tmp_dir, f"{key}_{j}.nii.gz")
        nib.save(slice, path_slice)

        # Append the 2D slice to the dict
        data.append({key: path_slice})
        
    logger.info("Image loaded!")
    return data


def inference_transforms(pixdim=(0.1,0.1)):
    transforms_list = [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=pixdim, align_corners=True, mode=("bilinear"), dtype=np.float32, scale_extent=True),
        NormalizeIntensityd(keys=["image"]),
    ]
    return Compose(transforms_list)

def load_model_monai_from_config(path_model):
    """
    Load a MONAI model from a configuration file.

    Args:
        path_model (str): Path to the model configuration file.

    Returns:
        torch.nn.Module: The loaded model.
    """
    device = torch.device("cpu")

    # Load the model configuration file
    config_path = os.path.join(path_model, "config.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = UNet(
                spatial_dims = config["UNet"]["spatial_dims"],
                in_channels = config["UNet"]["in_channels"],
                out_channels = config["UNet"]["out_channels"],
                channels = tuple(config["UNet"]["channels"]),
                strides = tuple(config["UNet"]["strides"]),
                num_res_units = config["UNet"]["num_res_units"],
                dropout = config["UNet"]["dropout"],
                act = config["UNet"]["activation"],
                norm = config["UNet"]["normalisation"],
            )
    chkp_path = os.path.join(path_model, "best_model_loss.ckpt")
    checkpoint = torch.load(chkp_path, map_location=torch.device(device))["state_dict"]

    for key in list(checkpoint.keys()):
        if 'net.' in key:
            checkpoint[key.replace('net.', '')] = checkpoint[key]
            del checkpoint[key]

    # load the trained model weights
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model

def postprocessing(batch, test_post_pred):

    # NOTE: monai's models do not normalize the output, so we need to do it manually
    if bool(F.relu(batch["pred"]).max()):
        batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max()
    else:
        batch["pred"] = F.relu(batch["pred"])

    post_test_out = [test_post_pred(i) for i in decollate_batch(batch)]

    pred = post_test_out[0]["pred"].cpu().detach().numpy()
    pred_wm = pred[0,:,:]
    pred_gm = pred[1,:,:]

    thr = 0.1
    pred_wm[pred_wm < thr] = 0
    pred_gm[pred_gm < thr] = 0

    remove = True
    if remove:
        #remove tiny blobs
        min_size = 100
        pred_wm = remove_smalls(pred_wm, min_size)
        pred_gm = remove_smalls(pred_gm, min_size)

    return pred_wm, pred_gm

def remove_smalls(predictions, min_size=10):
    """Remove the objects that have a size inferior to min_size

    Args:
        predictions (ndarray or nibabel object): Input segmentation. Image could be 2D or 3D.
        min_size (int): Minimum size of the objects to keep.

    Returns:
        ndarray or nibabel (same object as the input).
    """
    labels, num_labels = ndimage.label(predictions)
    sizes = ndimage.sum(predictions, labels, range(num_labels + 1))
    mask_size = sizes < min_size
    remove_pixel = mask_size[labels]
    predictions[remove_pixel] = 0

    return predictions

def tmp_create(basename):
    """Create temporary folder and return its path
    """
    prefix = f"sct_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{basename}_"
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    logger.info(f"Creating temporary folder ({tmpdir})")
    return tmpdir

def extract_fname(fpath):
    """
    Split a full path into a parent folder component, filename stem and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    parent, filename = os.path.split(fpath)
    if filename.endswith(".nii.gz"):
        stem, ext = filename[:-7], ".nii.gz"
    else:
        stem, ext = os.path.splitext(filename)
    return parent, stem, ext

# ===========================================================================
#                           Inference method
# ===========================================================================
def segment_monai_2d(path_img, tmpdir, predictor):
    
    # Copy the file to the temporary directory using shutil.copyfile
    path_img_tmp = os.path.join(tmpdir, os.path.basename(path_img))
    shutil.copyfile(path_img, path_img_tmp)
    logger.info(f'Copied {path_img} to {path_img_tmp}')

    # define pixdim for resampling  
    pixdim = (0.1, 0.1)

    # define the dataset and dataloader
    test_loader, test_post_pred, orig_image = prepare_data(path_img_tmp, pixdim=pixdim)
    
    # Run MONAI prediction
    print('Starting inference...')
    start = time()
    slices_wm=[]
    slices_gm=[]

    # run inference
    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            test_input = batch["image"].to(torch.device("cpu"))
            batch["pred"] = predictor(test_input)

            pred_wm, pred_gm = postprocessing(batch, test_post_pred)

            slices_wm.append(pred_wm)
            slices_gm.append(pred_gm)

        # slices = list(reversed(slices))  

        x,y,z = nib.aff2axcodes(orig_image.affine)

        # Apply the reorientation
        # TODO: use sct function to reorient the image
        if x == 'S' or x == 'I':
            axis = 0
        elif y == 'S' or y == 'I':
            axis = 1
        elif z == 'S' or z == 'I':
            axis = 2
        else:
            raise ValueError("Unknown axis orientation")
        
        final_volume_wm = np.stack(slices_wm, axis=axis)
        final_nib_wm = nib.Nifti1Image(final_volume_wm, orig_image.affine, orig_image.header)

        final_volume_gm = np.stack(slices_gm, axis=axis)
        final_nib_gm = nib.Nifti1Image(final_volume_gm, orig_image.affine, orig_image.header)

        end = time()
        print('Inference done.')
        total_time = end - start
        print(f'Total inference time: {int(total_time // 60)} minute(s) {int(round(total_time % 60))} seconds')

        # this takes about 0.25s on average on a CPU
        # image saver class
        _, fname, ext = extract_fname(path_img)
        postfix_wm = "wm_seg"
        postfix_gm = "gm_seg"
        target_wm = f"_{postfix_wm}"
        target_gm = f"_{postfix_gm}"
        # pred_saver = SaveImage(
        #     output_dir=tmpdir, output_postfix=postfix, output_ext=ext,
        #     separate_folder=False, print_log=False)
        # save the prediction
        fname_wm_out = os.path.join(tmpdir, f"{fname}_{postfix_wm}{ext}")
        fname_gm_out = os.path.join(tmpdir, f"{fname}_{postfix_gm}{ext}")
        logger.info(f"Saving results to: {tmpdir}")
        #pred_saver(pred)
        nib.save(final_nib_wm, fname_wm_out)
        nib.save(final_nib_gm, fname_gm_out)

    return [fname_wm_out, fname_gm_out], [target_wm, target_gm]

def segment_volume(path_model, input_filenames, threshold = 0.5, remove_temp_files=False):
    
    create_net = load_model_monai_from_config 
    inference = segment_monai_2d

    net = create_net(path_model)

    im_lst, target_lst = [], []
    for fname_in in input_filenames:
        tmpdir = tmp_create(basename="sct_deepseg")

        # model may be multiclass, so the `inference` func should output a list of fnames and targets
        fnames_out, targets = inference(path_img=fname_in, tmpdir=tmpdir, predictor=net)
        for fname_out, target in zip(fnames_out, targets):
            # im_out = Image(fname_out)
            # if threshold is not None:
            #     im_out.data = binarize(im_out.data, threshold)
            # im_lst.append(im_out)
            im_lst.append(fname_out)

            target_lst.append(target)
        if remove_temp_files:
            shutil.rmtree(tmpdir)

    return im_lst, target_lst


def main():
    
    # path to the model
    path_model = "/home/ge.polymtl.ca/jemal/data_nvme_jemal/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg/monai/models/gmseg"

    # path to the input image
    path_img = "/home/ge.polymtl.ca/jemal/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg/sub-3902bottom/anat/sub-3902bottom_T2w.nii.gz"

    # segment the image
    segment_volume(path_model, [path_img])

if __name__ == "__main__":
    main()