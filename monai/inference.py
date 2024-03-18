import os
import numpy as np
from loguru import logger
import torch.nn.functional as F
import torch
import yaml
import nibabel as nib

from datetime import datetime

from time import time

from skimage.transform import resize

from scipy.ndimage import zoom

from monai.data import (DataLoader, decollate_batch)
from monai.transforms import (Compose,ScaleIntensity)
from monai.networks.nets import UNet, DynUNet
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss

from data import Inference2dDataset
from utils import keep_largest_object, create_indexed_dir, get_last_folder_id
from main import Model
from loss import AdapWingLoss, BoundaryDiceLoss
from metrics import compute_dice_score

# ===========================================================================
#                   Prepare temporary dataset for inference
# ===========================================================================
def prepare_data(config):
        
        root = config["paths"]["dataset"]
        
        image_dir = os.path.join(root, config["paths"]["sub_paths"]["img"][0]) 
        data = []

        stems = [f[:-7] for f in os.listdir(image_dir) if  f.endswith('.nii.gz')]
        for stem in stems:
            data.append({
                "image": os.path.join(image_dir, f"{stem}.nii.gz"),
            })

        #TODO: coder proprement
        nib_image = nib.load(data[0]['image'])
        volume = nib_image.get_fdata()
        affine = nib_image.affine
        
        # define inference_transforms
        transform = Compose([
                ScaleIntensity(),
            ])
        
        logger.info(f"Loading dataset: {root}")      
        dataset = Inference2dDataset(data = data, axis = 1, transform=transform)

        return dataset, volume, affine

# ===========================================================================
#                           Inference method
# ===========================================================================
def main():
    device = "gpu"
    gpu_id = 3

    # Setup the device
    if device == "gpu" and not torch.cuda.is_available():
        print("GPU not available, using CPU instead")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")

    if device == torch.device("cpu"):
        raise KeyboardInterrupt("CPU is not supported for this task")
    else:
        print(f"Using {device} for training")
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU:{torch.cuda.current_device()} to run inference")
    
    # load the config file
    config_file_path = 'config.yaml'
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    model_stem = "best_model_dice.ckpt"

    # define the mask type
    mask_types = ["gm","wm"]

    for mask_type in mask_types:
        logger.info(f"Running inference for {mask_type} mask")

        # define root path for finding datalists
        training_path = os.path.join(config["paths"]["results"], str(get_last_folder_id(config["paths"]["results"])), mask_type)
        results_path = create_indexed_dir(os.path.join(training_path, "inference"))
        chkp_path = os.path.join(training_path, "models", model_stem)

        # save terminal outputs to a file
        logger.add(os.path.join(results_path, "logs.txt"), rotation="10 MB", level="INFO")

        logger.info(f"Saving results to: {results_path}")
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)

        # define the dataset and dataloader
        test_ds, volume, affine = prepare_data(config)
        print(f"Len dataset: {len(test_ds)}")
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        # define list to collect the test metrics
        test_step_outputs = []

        # define list to collect the final volume
        final_volume = []
            
        # define the model and loss (not saved in the checkpoint)
       
        net = UNet(
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
        
        
        #loss_function = AdapWingLoss(theta=0.5, omega=8, alpha=2.1, epsilon=1, reduction="sum")
        #loss_function = AdapWingLoss()
        #loss_function = DiceCELoss(include_background=False)
        #loss_function = DiceLoss(include_background=True)
        #loss_function = DiceFocalLoss()
        loss_function = BoundaryDiceLoss(0.5)

        # load the trained model weights
        model = Model.load_from_checkpoint(chkp_path, net =net, loss_function = loss_function)
        model.to(device)
        model.eval()

        # iterate over the dataset and compute metrics
        with torch.no_grad():
            for i, batch in enumerate(test_loader):

                # compute time for inference per slice
                start_time = time()

                # get the test input
                test_input = batch["image"].to(device)

                # run inference  
                batch["pred"] = model(test_input)

                # take only the highest resolution prediction
                batch["pred"] = batch["pred"][0]

                # NOTE: monai's models do not normalize the output, so we need to do it manually
                if bool(F.relu(batch["pred"]).max()):
                    batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max() 
                else:
                    batch["pred"] = F.relu(batch["pred"])

                # postprocessing
                post_test_out = decollate_batch(batch)  

                pred = post_test_out[0]['pred'].cpu()
                pred = pred.numpy()

                # threshold the prediction to set all values below pred_thr to 0
                pred[pred < config["postprocessing"]["remove_noise"]["threshold"]] = 0
                keep_largest = True

                if keep_largest:
                    # keep only the largest connected component (to remove tiny blobs after thresholding)
                    #logger.info("Postprocessing: Keeping the largest connected component in the prediction")
                    pred = keep_largest_object(pred)
                
                #TODO: coder proprement
                slice_final_volume = resize(pred, (128,128), anti_aliasing=True)

                final_volume.append(slice_final_volume)
                    
                end_time = time()
                metrics_dict = {
                    "slice_id": i,
                    "inference_time_in_sec": round((end_time - start_time), 2),
                }
                test_step_outputs.append(metrics_dict)

            # save the final volume
            final_volume = np.array(final_volume)

            final_volume = np.moveaxis(final_volume, 1, 0)
            final_nib = nib.Nifti1Image(final_volume, affine)

            nib.save(final_nib, os.path.join(results_path, "segmentation.nii.gz"))

            # compute the average inference time
            sum_inference_time = np.stack([x["inference_time_in_sec"] for x in test_step_outputs]).sum()

            logger.info("========================================================")
            logger.info(f"      Inference Time per Subject: {sum_inference_time:.2f}s")
            logger.info("========================================================")

        # compute metrics
        groundtruth_path = "/home/ge.polymtl.ca/jemal/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg/derivatives/labels/sub-3902bottom/anat"
        if mask_type == "gm":
            groundtruth_path = os.path.join(groundtruth_path, "sub-3902bottom_T2w_gmseg_manual.nii.gz")
        else:
            groundtruth_path = os.path.join(groundtruth_path, "sub-3902bottom_T2w_wmseg_manual.nii.gz")
        groundtruth = nib.load(groundtruth_path).get_fdata()
        new_segmentation = final_volume
        ivadomed_seg_path = "/home/ge.polymtl.ca/jemal/data_nvme_jemal/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg/monai/results_ivadomed/"
        if mask_type == "gm":
            ivadomed_seg_path = os.path.join(ivadomed_seg_path, "sub-3902bottom_T2w_gmseg.nii.gz")
        else:
            ivadomed_seg_path = os.path.join(ivadomed_seg_path, "sub-3902bottom_T2w_wmseg.nii.gz")
        ivadomed_segmentation = nib.load(ivadomed_seg_path).get_fdata()

        # threshold the soft segmentations
        thr = 0.5
        groundtruth[groundtruth < groundtruth.max()*thr] = 0
        new_segmentation[new_segmentation < thr] = 0
        ivadomed_segmentation[ivadomed_segmentation < thr] = 0

        ivadomed_dice = compute_dice_score(ivadomed_segmentation, groundtruth)
        new_dice = compute_dice_score(new_segmentation, groundtruth)

        with open(os.path.join(results_path, 'metrics.txt'), 'a') as f:
            print('\n-------------- Metrics ----------------', file=f)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            print(f'{timestamp}', file=f)
            print(f'Ivadomed_dice: {ivadomed_dice}', file=f)
            print(f'New_dice: {new_dice}', file=f)
        
if __name__ == "__main__":
    main()