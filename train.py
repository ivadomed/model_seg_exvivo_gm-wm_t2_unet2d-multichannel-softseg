#%%
# Importing libraries
import os
import numpy as np
import yaml

from monai.networks.nets import UNet
from monai.transforms import Compose, ScaleIntensity
from monai.data import DataLoader 

import torch
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.transforms import Resize

from transforms import RandAffineRel
from data import Segmentation2dDataset
from loss import AdapWingLoss
from utils import create_model_dir

#%%
#should add logger

device = "gpu"
gpu_id = 1

config_file_path = 'config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

class UNetTrainer:
    def __init__(self, config, mask_type, device="gpu", gpu_id=0):
        if __name__ == "__main__":
            self.device = self._setup_device(device, gpu_id)
        else:
            self.device = device
        self.root_path = config["paths"]["dataset"]
        self.mask_type = mask_type
        self.batch_size = config["training"]["batch_size"]
        self.num_epochs = config["training"]["num_epochs"]
        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.early_stopping_epsilon = config["training"]["early_stopping_epsilon"]
        self.learning_rate = config["training"]["learning_rate"]
        self.eta_min = config["training"]["eta_min"]   
        self.axis = config["data"]["slice_axis"] 
        self._setup_data(config["paths"]["sub_paths"], mask_type)
        self._setup_transforms(config["transformations"])
        self.train_set_size = config["training"]["train_set_size"]
        self.val_set_size = config["training"]["val_set_size"]
        self._setup_dataloaders()
        self.model = self._create_model(config["UNet"])
        self.criterion = AdapWingLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=self.eta_min)
        self.results_dir = config["paths"]["results"]

    def _setup_device(self, device, gpu_id):
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
            print(torch.cuda.current_device())
        return device

    def _setup_data(self, sub_paths_dict, mask_type):
        if mask_type == "wm":
            ext = "wmseg_manual"
        elif mask_type == "gm":
            ext = "gmseg_manual"
        else:
            raise ValueError("Invalid mask type")
        
        image_dir = [os.path.join(self.root_path, sub_path) for sub_path in sub_paths_dict["img"]]
        masks_dir = [os.path.join(self.root_path, sub_path) for sub_path in sub_paths_dict["masks"]]

        self.data = []

        for i in range(len(image_dir)):
            stems = [f[:-7] for f in os.listdir(image_dir[i]) if  f.endswith('.nii.gz')]
            for stem in stems:
                if not os.path.exists(os.path.join(masks_dir[i], f"{stem}_{ext}.nii.gz")):
                    raise ValueError(f"No {mask_type} mask found for {stem}")
                else :
                    self.data.append({
                        "image": os.path.join(image_dir[i], f"{stem}.nii.gz"),
                        "mask": os.path.join(masks_dir[i], f"{stem}_{ext}.nii.gz")
                    })
    
    def _setup_transforms(self, transforms_dict):
        proba, affine_degrees, affine_translation = transforms_dict["randaffine"]["proba"], transforms_dict["randaffine"]["degrees"], transforms_dict["randaffine"]["translation"]

        self.transform = Compose([
                RandAffineRel(prob=proba, affine_degrees=affine_degrees, affine_translation=affine_translation),
                ScaleIntensity(),
            ])
        
    def _setup_dataloaders(self):
        
        dataset = Segmentation2dDataset(data = self.data, axis = self.axis, transform=self.transform)

        image, mask = dataset[0]["image"], dataset[0]["mask"]
        
        train_dataset, val_dataset = random_split(dataset, [self.train_set_size, self.val_set_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

    def _create_model(self, model_params):
        model = UNet(
            spatial_dims = model_params["spatial_dims"],
            in_channels = model_params["in_channels"],
            out_channels = model_params["out_channels"],
            channels = tuple(model_params["channels"]),
            strides = tuple(model_params["strides"]),
            num_res_units = model_params["num_res_units"],
            dropout = model_params["dropout"],
            act = model_params["activation"],
            norm = model_params["normalisation"],
        ).to(self.device)
        return model

    def train(self):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch in self.train_loader:
                images, masks = batch['image'], batch['mask']
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            self.scheduler.step()
            epoch_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

            val_loss = self._validate()

            if val_loss < best_val_loss - self.early_stopping_epsilon:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Check early stopping condition
            if epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    def _validate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                images, masks = batch['image'], batch['mask']
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()

        average_val_loss = val_loss / len(self.val_loader)
        return average_val_loss

    def save_model(self, name = "model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg"):
        if self.mask_type == "wm":
            ext = "wmseg"
        elif self.mask_type == "gm":
            ext = "gmseg"
        else:
            raise ValueError("Invalid mask type")

        name = f"{name}_{ext}.pth"

        model_dir = create_model_dir(self.results_dir)
        model_path = os.path.join(model_dir, name)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at {model_path}")
        config_path = os.path.join(model_dir, "config.yaml")
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
        print(f"Config file saved at {config_path}")

if __name__ == "__main__":
    root_path = config["paths"]["dataset"]
    trainer = UNetTrainer(config=config, device=device, gpu_id=gpu_id)
    trainer.train(mask_type="mask_wm")
    trainer.save_model(mask_type="mask_wm")

               
