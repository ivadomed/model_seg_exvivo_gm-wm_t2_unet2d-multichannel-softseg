import os
import numpy as np
from loguru import logger
import torch.nn.functional as F
import torch
import yaml
import nibabel as nib
import matplotlib.pyplot as plt

from datetime import datetime

from time import time

from skimage.transform import resize

from monai.data import (DataLoader, decollate_batch)
from monai.transforms import (Compose, EnsureTyped, Invertd)
from monai.networks.nets import UNet

from data import Inference2dDataset
from utils import keep_largest_object, create_indexed_dir, get_last_folder_id
from main import Model
from loss import AdapWingLoss
from metrics import compute_dice_score
from transforms import inference_transforms

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
        
        # define inference_transforms
        transform = inference_transforms(pixdim=config["transformations"]["resampling"]["pixdim"],) 
        
        logger.info(f"Loading dataset: {root}")      
        dataset = Inference2dDataset(data = data, transform=transform)

        test_post_pred = Invertd(keys=["pred"], 
                                transform=transform, 
                                orig_keys=["image"])

        return dataset, test_post_pred, nib_image

# ===========================================================================
#                           Inference method
# ===========================================================================
def main():
    device = "gpu"
    gpu_id = 1

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
        test_ds, test_post_pred, nib_image = prepare_data(config)

        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        # define list to collect the test metrics
        test_step_outputs = []

        # define list to collect the final volume
        #original_volume = nib_image.get_fdata()
        slices = []
            
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
        
        # define the loss function
        loss_function = AdapWingLoss(theta=0.5, omega=8, alpha=2.1, epsilon=1, reduction="sum")

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
                #post_test_out = decollate_batch(batch)  
                post_test_out =  [test_post_pred(i) for i in decollate_batch(batch)]

                pred_nifti = post_test_out[0]['pred']
                pred = pred_nifti.get_fdata()

                #remove the channel dimension
                pred = pred.squeeze(0)

                # threshold the prediction to set all values below pred_thr to 0
                pred[pred < config["postprocessing"]["remove_noise"]["threshold"]] = 0
                keep_largest = True

                if keep_largest:
                    # keep only the largest connected component (to remove tiny blobs after thresholding)
                    #logger.info("Postprocessing: Keeping the largest connected component in the prediction")
                    pred = keep_largest_object(pred)

                slices.append(pred)
                    
                end_time = time()
                metrics_dict = {
                    "slice_id": i,
                    "inference_time_in_sec": round((end_time - start_time), 2),
                }
                test_step_outputs.append(metrics_dict)
     
            slices = list(reversed(slices))  
            final_volume = np.stack(slices, axis=2)
            final_volume = final_volume.swapaxes(2,1)
            final_volume = np.rot90(final_volume, k=1, axes=(0, 2)) 

            final_nib = nib.Nifti1Image(final_volume, nib_image.affine, nib_image.header)

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

        # save slices for visualization
        idx_list=[100, 120,140,160,180,200]

        fig, ax = plt.subplots(4,len(idx_list), figsize=(20, 18))
        titles = ["Original", "Ground Truth", "New Segmentation", "Ivadomed Segmentation"]

        for i,idx in enumerate(idx_list): 
            ax[0][i].imshow(nib_image.get_fdata()[:,idx,:], cmap="gray")
            ax[1][i].imshow(nib_image.get_fdata()[:,idx,:], cmap="gray")
            ax[1][i].imshow(groundtruth[:,idx,:], cmap="copper" , alpha=0.6)
            ax[2][i].imshow(nib_image.get_fdata()[:,idx,:], cmap="gray")
            ax[2][i].imshow(new_segmentation[:,idx,:], cmap="copper" , alpha=0.6)
            ax[3][i].imshow(nib_image.get_fdata()[:,idx,:], cmap="gray")
            ax[3][i].imshow(ivadomed_segmentation[:,idx,:], cmap="copper" , alpha=0.6)

        for a in ax.flatten():
            a.axis('off')

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.88, hspace=0.4, wspace=0.2)

        for row, title in enumerate(titles):
            fig.text(0.5, 0.91-(row*0.222), title, ha='center', va='center', fontsize=18, fontweight='heavy') 

        fig.savefig(os.path.join(results_path, "slices.png"), dpi=300, bbox_inches='tight')

        # threshold the soft segmentations
        thr = 0.5
        # groundtruth[groundtruth < groundtruth.max()*thr] = 0
        # new_segmentation[new_segmentation < thr] = 0
        # ivadomed_segmentation[ivadomed_segmentation < thr] = 0
        groundtruth = (groundtruth > thr).astype(int)
        new_segmentation = (new_segmentation > thr).astype(int)
        ivadomed_segmentation = (ivadomed_segmentation > thr).astype(int)

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