#%%

# Import the required libraries
import torch
import os
import numpy as np
import yaml
from PIL import Image

from monai.networks.nets import UNet
from monai.transforms import Compose, ScaleIntensity

from torch.utils.data import DataLoader

from skimage.transform import resize

from transforms import RandAffineRel
from data import Inference2dDataset
from utils import check_existing_model, create_seg_dir
from train import UNetTrainer
from postprocessing import remove_noise, remove_small

# Define inference parameters
device = "gpu"
gpu_id = 1

config_file_path = 'config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

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

# Load a model, or train a new one
model_path = check_existing_model(config["paths"]["results"])

if model_path is None:
    print("No model found, training a new one")
    trainer = UNetTrainer(config, mask_type="wm", device = device, gpu_id = gpu_id)
    trainer.train()
    trainer.save_model()
    model = trainer.model
else:
    print(f"Loading model from {model_path}")
    config_file_path = os.path.join(os.path.dirname(model_path), "config.yaml")
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
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
    ).to(device)
    model.load_state_dict(torch.load(model_path))

result_dir = create_seg_dir(config["paths"]["results"])

transform = Compose([
    RandAffineRel(
        prob= config["transformations"]["randaffine"]["proba"], 
        affine_degrees = config["transformations"]["randaffine"]["degrees"], 
        affine_translation = tuple(config["transformations"]["randaffine"]["translation"])
        ),
    ScaleIntensity()
])

# Dataset and dataloader
root_path = config["paths"]["dataset"]
images_rel_dir =  config["paths"]["sub_paths"]["img"]
images_dir = os.path.join(root_path, images_rel_dir[0]) 
data = []

fnames = [f for f in os.listdir(images_dir) if  f.endswith('.nii.gz')]
for fname in fnames:
    data.append({
        "image": os.path.join(images_dir, fname)
    })

import nibabel as nib
nib_image = nib.load(data[0]['image'])
affine = nib_image.affine

segmentation_dataset = Inference2dDataset(data=data, axis = config["data"]["slice_axis"], transform=transform)
segmentation_loader = DataLoader(segmentation_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
#%%
# Run inference
segmented_outputs = []
with torch.no_grad():  # No need to track gradients
    for batch in segmentation_loader:
        images = batch['image']
        images = images.to(device)  # Move images to the appropriate device
        outputs = model(images)
        # Store or process your outputs here
        segmented_outputs.append({"image": images, "seg": outputs.cpu()})  

# Post-processing
final_volume = []   
for image_index, sample in enumerate(segmented_outputs):
    for i, seg_map in enumerate(sample["seg"][:,0,:,:]):  
        seg_map_np = seg_map.cpu().numpy()
        seg_map_np = remove_noise(seg_map_np, config["postprocessing"]["remove_noise"]["threshold"])
        seg_map_np = remove_small(seg_map_np, config["postprocessing"]["remove_small"]["min_size"])

        segmented_outputs[image_index]["seg"][i,0,:,:] = torch.tensor(seg_map_np)
        
        seg_map_np = resize(seg_map_np, (128,128), anti_aliasing=True)
        final_volume.append(seg_map_np)

final_volume = np.array(final_volume)
final_nib = nib.Nifti1Image(final_volume, affine)

nib.save(final_nib, os.path.join(result_dir, "segmentation.nii.gz"))

# Save the segmentation maps
for image_index, sample in enumerate(segmented_outputs): 
    images = sample["image"][:,0,:,:]
    segmentation_map = sample["seg"][:,0,:,:]
    for i, (img, seg_map) in enumerate(zip(images, segmentation_map)):       
        seg_map_np = seg_map.cpu().numpy()       
        # Convert to an image (using PIL) and save
        seg = Image.fromarray(np.uint8(seg_map_np * 255))  # Scale values if needed
        seg.save(os.path.join(result_dir, f'{image_index}_{i}_seg.png'))
        img = Image.fromarray(np.uint8(img.cpu().numpy() * 255))
        img.save(os.path.join(result_dir, f'{image_index}_{i}_img.png'))
        #print(f"Segmentation map {i} for batch {image_index} on {len(segmented_outputs)} saved to {result_dir}")
