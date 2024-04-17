import os
from datetime import datetime
from loguru import logger
import yaml
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

import torch.optim as optim
from torch.utils.data import random_split
from pytorch_lightning.loggers import WandbLogger

from monai.utils import set_determinism
from monai.networks.nets import UNet
from monai.data import (DataLoader, decollate_batch)
from monai.transforms import (Compose, EnsureType)

from utils import plot_slices, check_empty_patch, create_indexed_dir, get_last_folder_id
from metrics import dice_score
from loss import AdapWingLoss, BoundaryDiceLoss, BoundaryAdapWingLoss
from transforms import training_transforms_multiclass
from data import Segmentation2dDatasetMulticlass


# create a "model"-agnostic class with PL to use different models
class Model(pl.LightningModule):
    def __init__(self, config, data_root, optimizer_class, loss_function,  net, seed, debug = False, exp_id=None, results_path=None):
        super().__init__() 
        self.config = config
        self.save_hyperparameters(ignore=['net', 'loss_function'])
        #self.save_hyperparameters()

        self.root = data_root
        self.net = net
        self.lr = config["training"]["learning_rate"]
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.seed = seed
        self.debug = debug
        self.save_exp_id = exp_id
        self.results_path = results_path

        self.best_val_dice, self.best_val_loss_epoch, self.best_val_loss_epoch = 0, 0, 0
        self.best_val_loss = float("inf")

        self.spacing = config["transformations"]["resampling"]["pixdim"]
        self.voxel_cropping_size = self.inference_roi_size = (200,200)

        # define post-processing transforms for validation, nothing fancy just making sure that it's a tensor (default)
        self.val_post_pred = Compose([EnsureType()]) 
        self.val_post_label = Compose([EnsureType()])

        # define evaluation metric
        self.soft_dice_metric = dice_score

        # temp lists for storing outputs from training, validation, and testing
        self.train_step_outputs = []
        self.val_step_outputs = []


    # --------------------------------
    # FORWARD PASS
    # --------------------------------
    def forward(self, x):
        
        out = self.net(x)  
        # # NOTE: MONAI's models only output the logits, not the output after the final activation function
        # # https://docs.monai.io/en/0.9.0/_modules/monai/networks/nets/unetr.html#UNETR.forward refers to the 
        # # UnetOutBlock (https://docs.monai.io/en/0.9.0/_modules/monai/networks/blocks/dynunet_block.html#UnetOutBlock) 
        # # as the final block applied to the input, which is just a convolutional layer with no activation function
        # # Hence, we are used Normalized ReLU to normalize the logits to the final output
        # normalized_out = F.relu(out) / F.relu(out).max() if bool(F.relu(out).max()) else F.relu(out)
        return out  # returns logits

    # --------------------------------
    # DATA PREPARATION
    # --------------------------------   
    def prepare_data(self):
        # set deterministic training for reproducibility
        set_determinism(seed=self.seed)

        image_dir = [os.path.join(self.root, sub_path) for sub_path in self.config["paths"]["sub_paths"]["img"]]
        masks_dir = [os.path.join(self.root, sub_path) for sub_path in self.config["paths"]["sub_paths"]["masks"]]

        data = []

        for i in range(len(image_dir)):
            stems = [f[:-7] for f in os.listdir(image_dir[i]) if  f.endswith('.nii.gz')]
            for stem in stems:
                if not os.path.exists(os.path.join(masks_dir[i], f"{stem}_wmseg_manual.nii.gz")):
                    raise ValueError(f"No wm mask found for {stem}")
                elif not os.path.exists(os.path.join(masks_dir[i], f"{stem}_gmseg_manual.nii.gz")):
                    raise ValueError(f"No gm mask found for {stem}")
                else :
                    data.append({
                        "image": os.path.join(image_dir[i], f"{stem}.nii.gz"),
                        "mask_wm": os.path.join(masks_dir[i], f"{stem}_wmseg_manual.nii.gz"),
                        "mask_gm": os.path.join(masks_dir[i], f"{stem}_gmseg_manual.nii.gz"),
                    })
        
        # define training and validation transforms
        affine_degrees, affine_translation = self.config["transformations"]["randaffine"]["degrees"], self.config["transformations"]["randaffine"]["translation"]

        transform = training_transforms_multiclass(pixdim=self.spacing, translate_range=affine_translation, rotate_range=affine_degrees)
        
        logger.info(f"Loading dataset: {self.root}")      
        dataset = Segmentation2dDatasetMulticlass(data = data, transform=transform)
        self.train_dataset, self.val_dataset = random_split(dataset, [self.config["training"]["train_set_size"], self.config["training"]["val_set_size"]])
        

    # --------------------------------
    # DATA LOADERS
    # --------------------------------
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["training"]["batch_size"], shuffle=True, num_workers=8 , pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config["training"]["batch_size"], shuffle=False, num_workers=8, pin_memory=True)
    
    # --------------------------------
    # OPTIMIZATION
    # --------------------------------
    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.parameters(), lr=self.config["training"]["learning_rate"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["training"]["num_epochs"], eta_min=self.config["training"]["eta_min"])
        return [optimizer], [scheduler]


    # --------------------------------
    # TRAINING

    # --------------------------------
    def training_step(self, batch):

        inputs, masks_wm, masks_gm = batch["image"], batch["mask_wm"], batch["mask_gm"]  

        # check if any label image patch is empty in the batch
        if check_empty_patch(masks_wm) is None or check_empty_patch(masks_gm) is None:
            #print(f"Empty label patch found. Skipping training step ...")
            return None

        out = self.forward(inputs)   # logits
        output_wm, output_gm = out.split(1, dim=1)
        #print(f"labels.shape: {masks.shape} \t output.shape: {output.shape}")
        
        # calculate training loss   
        loss_wm = self.loss_function(output_wm, masks_wm)
        loss_gm = self.loss_function(output_gm, masks_gm)

        loss = loss_wm + loss_gm

        # get probabilities from logits
        output_wm = F.relu(output_wm) / F.relu(output_wm).max() if bool(F.relu(output_wm).max()) else F.relu(output_wm)
        output_gm = F.relu(output_gm) / F.relu(output_gm).max() if bool(F.relu(output_gm).max()) else F.relu(output_gm)

        # calculate train dice
        train_soft_dice_wm = self.soft_dice_metric(output_wm, masks_wm)
        train_soft_dice_gm = self.soft_dice_metric(output_gm, masks_gm)

        train_soft_dice = (train_soft_dice_wm + train_soft_dice_gm) / 2 

        metrics_dict = {
            "loss": loss.cpu(),
            "train_soft_dice": train_soft_dice.detach().cpu(),
            "train_number": len(inputs),
            "train_image": inputs[0].detach().cpu().squeeze(),
            "train_gt_wm": masks_wm[0].detach().cpu().squeeze(),
            "train_gt_gm": masks_gm[0].detach().cpu().squeeze(),
            "train_pred_wm": output_wm[0].detach().cpu().squeeze(),
            "train_pred_gm": output_gm[0].detach().cpu().squeeze(),
        }
        self.train_step_outputs.append(metrics_dict)

        return metrics_dict

    def on_train_epoch_end(self):

        if self.train_step_outputs == []:
            # means the training step was skipped because of empty input patch
            return None
        else:
            train_loss, train_soft_dice = 0, 0
            num_items = len(self.train_step_outputs)
            for output in self.train_step_outputs:
                train_loss += output["loss"].item()
                train_soft_dice += output["train_soft_dice"].item()
            
            mean_train_loss = (train_loss / num_items)
            mean_train_soft_dice = (train_soft_dice / num_items)

            wandb_logs = {
                "train_soft_dice": mean_train_soft_dice, 
                "train_loss": mean_train_loss,
            }
            self.log_dict(wandb_logs)

            # plot the training images
            fig_wm = plot_slices(image=self.train_step_outputs[0]["train_image"],
                            gt=self.train_step_outputs[0]["train_gt_wm"],
                            pred=self.train_step_outputs[0]["train_pred_wm"],
                            debug=self.debug)

            fig_gm = plot_slices(image=self.train_step_outputs[0]["train_image"],
                                gt=self.train_step_outputs[0]["train_gt_gm"],
                                pred=self.train_step_outputs[0]["train_pred_gm"],
                                debug=self.debug)

            wandb.log({"validation images white matter": wandb.Image(fig_wm)})
            wandb.log({"validation images grey matter": wandb.Image(fig_gm)})

            # free up memory
            self.train_step_outputs.clear()
            wandb_logs.clear()
            plt.close(fig_wm)
            plt.close(fig_gm)


    # --------------------------------
    # VALIDATION
    # --------------------------------    
    def validation_step(self, batch, batch_idx):
        
        inputs, masks_wm, masks_gm = batch["image"], batch["mask_wm"], batch["mask_gm"]  

        # NOTE: this calculates the loss on the entire image after sliding window
        # outputs = sliding_window_inference(inputs, self.inference_roi_size, mode="gaussian",
        #                                    sw_batch_size=4, predictor=self.forward, overlap=0.5,) 
        # outputs shape: (B, C, <original H x W x D>)
        out = self.forward(inputs)   
        outputs_wm, outputs_gm = out.split(1, dim=1) 

        # calculate validation loss
        loss_wm = self.loss_function(outputs_wm, masks_wm)
        loss_gm = self.loss_function(outputs_gm, masks_gm)

        loss = loss_wm + loss_gm

        # get probabilities from logits
        outputs_wm = F.relu(outputs_wm) / F.relu(outputs_wm).max() if bool(F.relu(outputs_wm).max()) else F.relu(outputs_wm)
        
        # post-process for calculating the evaluation metric
        post_outputs_wm = [self.val_post_pred(i) for i in decollate_batch(outputs_wm)]
        post_labels_wm = [self.val_post_label(i) for i in decollate_batch(masks_wm)]
        val_soft_dice_wm = self.soft_dice_metric(post_outputs_wm[0], post_labels_wm[0])
        post_outputs_gm = [self.val_post_pred(i) for i in decollate_batch(outputs_gm)]
        post_labels_gm = [self.val_post_label(i) for i in decollate_batch(masks_gm)]
        val_soft_dice_gm = self.soft_dice_metric(post_outputs_gm[0], post_labels_gm[0])

        val_soft_dice = (val_soft_dice_wm + val_soft_dice_gm) / 2

        # calculate hard dice
        hard_preds_wm, hard_labels_wm = (post_outputs_wm[0].detach() > 0.5).float(), (post_labels_wm[0].detach() > 0.5).float()
        val_hard_dice_wm = self.soft_dice_metric(hard_preds_wm, hard_labels_wm)
        hard_preds_gm, hard_labels_gm = (post_outputs_gm[0].detach() > 0.5).float(), (post_labels_gm[0].detach() > 0.5).float()
        val_hard_dice_gm = self.soft_dice_metric(hard_preds_gm, hard_labels_gm)

        val_hard_dice = (val_hard_dice_wm + val_hard_dice_gm) / 2

        metrics_dict = {
            "val_loss": loss.detach().cpu(),
            "val_soft_dice": val_soft_dice.detach().cpu(),
            "val_hard_dice": val_hard_dice.detach().cpu(),
            "val_number": len(post_outputs_wm),
            "val_image": inputs[0].detach().cpu().squeeze(),
            "val_gt_wm": masks_wm[0].detach().cpu().squeeze(),
            "val_gt_gm": masks_gm[0].detach().cpu().squeeze(),
            "val_pred_wm": post_outputs_wm[0].detach().cpu().squeeze(),
            "val_pred_gm": post_outputs_gm[0].detach().cpu().squeeze(),
        }
        self.val_step_outputs.append(metrics_dict)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_soft_dice', val_soft_dice, prog_bar=True)
        self.log('val_hard_dice', val_hard_dice, prog_bar=True)
        
        return metrics_dict

    def on_validation_epoch_end(self):

        val_loss, num_items, val_soft_dice, val_hard_dice = 0, 0, 0, 0
        for output in self.val_step_outputs:
            val_loss += output["val_loss"].sum().item()
            val_soft_dice += output["val_soft_dice"].sum().item()
            val_hard_dice += output["val_hard_dice"].sum().item()
            num_items += output["val_number"]
        
        mean_val_loss = (val_loss / num_items)
        mean_val_soft_dice = (val_soft_dice / num_items)
        mean_val_hard_dice = (val_hard_dice / num_items)
                
        wandb_logs = {
            "val_soft_dice": mean_val_soft_dice,
            "val_hard_dice": mean_val_hard_dice,
            "val_loss": mean_val_loss,
        }

        # save the best model based on validation dice score
        if mean_val_soft_dice > self.best_val_dice:
            self.best_val_dice = mean_val_soft_dice
            self.best_val_dice_epoch = self.current_epoch
        
        # save the best model based on validation CSA loss
        if mean_val_loss < self.best_val_loss:    
            self.best_val_loss = mean_val_loss
            self.best_val_loss_epoch = self.current_epoch

        logger.info(
            f"\nCurrent epoch: {self.current_epoch}"
            f"\nAverage Soft Dice (VAL): {mean_val_soft_dice:.4f}"
            f"\nAverage Hard Dice (VAL): {mean_val_hard_dice:.4f}"
            f"\nAverage AdapWing Loss (VAL): {mean_val_loss:.4f}"
            f"\nBest Average Soft Dice: {self.best_val_dice:.4f} at Epoch: {self.best_val_dice_epoch}"
            f"\nBest Average AdapWing Loss: {self.best_val_loss:.4f} at Epoch: {self.best_val_loss_epoch}"
            f"\n----------------------------------------------------")
        

        # log on to wandb
        self.log_dict(wandb_logs)

        # # plot the validation images
        fig_wm = plot_slices(image=self.val_step_outputs[0]["val_image"],
                          gt=self.val_step_outputs[0]["val_gt_wm"],
                          pred=self.val_step_outputs[0]["val_pred_wm"],)

        fig_gm = plot_slices(image=self.val_step_outputs[0]["val_image"],
                            gt=self.val_step_outputs[0]["val_gt_gm"],
                            pred=self.val_step_outputs[0]["val_pred_gm"],)

        wandb.log({"validation images white matter": wandb.Image(fig_wm)})
        wandb.log({"validation images grey matter": wandb.Image(fig_gm)})

        # free up memory
        self.val_step_outputs.clear()
        wandb_logs.clear()
        plt.close(fig_wm)
        plt.close(fig_gm)
        
        # return {"log": wandb_logs}


# --------------------------------
# MAIN
# --------------------------------
def main(seed, check_val_every_n_epochs, max_epochs):
    # Setting the seed
    pl.seed_everything(seed, workers=True)
    # ====================================================================================================

    # Load the configuration file
    config_file_path = '/home/ge.polymtl.ca/jemal/data_nvme_jemal/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg/monai/config.yaml'
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # define root path for finding datalists
    dataset_root = config["paths"]["dataset"] 
    training_dir_masks = create_indexed_dir(config["paths"]["results"])

    # define optimizer
    if config["training"]["optimizer"] in ["adam", "Adam"]:
        optimizer_class = torch.optim.Adam
    elif config["training"]["optimizer"] in ["SGD", "sgd"]:
        optimizer_class = torch.optim.SGD

    model_type = "UNet"

    # create a directory to save the training results
    training_dir = os.path.join(training_dir_masks, "multiclass")

    # define models
    if model_type in ["unet", "UNet"]:
        # define image size to be fed to the model
        
        # define model
        net = UNet(
            spatial_dims = config["UNet"]["spatial_dims"],
            in_channels = config["UNet"]["in_channels"],
            out_channels = 2,
            channels = tuple(config["UNet"]["channels"]),
            strides = tuple(config["UNet"]["strides"]),
            num_res_units = config["UNet"]["num_res_units"],
            dropout = config["UNet"]["dropout"],
            act = config["UNet"]["activation"],
            norm = config["UNet"]["normalisation"],
        )

        lr = config["training"]["learning_rate"]
        bs = config["training"]["batch_size"]

    # save output to a log file
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)
    logger.add(os.path.join(training_dir, "logs.txt"), rotation="10 MB", level="INFO")

    loss_func = BoundaryAdapWingLoss(boundary_weight=0.3, theta=0.5, omega=8, alpha=2.1, epsilon=1, reduction="sum")

    logger.info(f"Using BoundaryAdapWingLoss with ratio {loss_func.boundary_weight*100}/{100 - loss_func.boundary_weight*100}; theta={loss_func.theta}, omega={loss_func.omega}, alpha={loss_func.alpha}, epsilon={loss_func.epsilon}!")

    # define callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, 
                                                patience=config["training"]["early_stopping_patience"], verbose=False, mode="min")

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # Training
    # to save the best model on validation
    save_path = os.path.join(training_dir, "models")

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # to save the results/model predictions 
    results_path = os.path.join(training_dir, "results_train")
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    inference_path = os.path.join(training_dir, "inference")
    if not os.path.exists(inference_path):
        os.makedirs(inference_path, exist_ok=True)

    # i.e. train by loading weights from scratch
    pl_model = Model(config = config, data_root= dataset_root, 
                        optimizer_class= optimizer_class, loss_function= loss_func, net= net, 
                        seed= seed, results_path=results_path)
            
    # saving the best model based on validation loss
    logger.info(f"Saving best model to {save_path}!")
    checkpoint_callback_loss = pl.callbacks.ModelCheckpoint(
        dirpath=save_path, filename='best_model_loss', monitor='val_loss', 
        save_top_k=1, mode="min", save_last=True, save_weights_only=False)
    
    # saving the best model based on soft validation dice score
    checkpoint_callback_dice = pl.callbacks.ModelCheckpoint(
        dirpath=save_path, filename='best_model_dice', monitor='val_soft_dice', 
        save_top_k=1, mode="max", save_last=False, save_weights_only=True)
    
    logger.info(f" Starting training from scratch! ")
    # wandb logger
    grp = f"monai_ivado_{model_type}" if model_type in ["unet", "UNet"] else f"monai_{model_type}"
    folder_id = get_last_folder_id(config["paths"]["results"])
    exp_logger = WandbLogger(
                        name=f"training_{folder_id}_multiclass_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                        save_dir=save_path,
                        group=grp,
                        log_model=True, # save best model using checkpoint callback
                        project='ivadomed2monai',
                        entity='',
                        config=config) 

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        devices=1, accelerator="gpu", # strategy="ddp",
        logger=exp_logger,
        callbacks=[checkpoint_callback_loss, checkpoint_callback_dice, lr_monitor, early_stopping],
        check_val_every_n_epoch=check_val_every_n_epochs,
        max_epochs= max_epochs, 
        precision=32,   
        # deterministic=True,
        enable_progress_bar= True, 
        profiler="simple",)     # to profile the training time taken for each step

    # Train!
    trainer.fit(pl_model)        
    logger.info(f"Training Done!")

    
    # Saving training script to wan db
    wandb.save("main.py")
    wandb.save("transforms.py")

    # closing the current wandb instance so that a new one is created for the next fold
    wandb.finish()    


    with open(os.path.join(results_path, 'val_metrics.txt'), 'a') as f:
        print('\n-------------- Val Metrics ----------------', file=f)
        print(f"\nSeed Used: {seed}", file=f)
        lr = config["training"]["learning_rate"]
        bs = config["training"]["batch_size"]
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        print(f"\nlr={lr}_bs={bs}_{timestamp}", file=f)
        print(f"\npatch_size={pl_model.voxel_cropping_size}", file=f)
        print('\n-------------- Loss ----------------', file=f)
        print("Best Validation Loss --> %0.3f at Epoch: %0.3f" % (pl_model.best_val_loss, pl_model.best_val_loss_epoch), file=f)

        print('\n-------------- Dice Score ----------------', file=f)
        print("Best Dice Score --> %0.3f at Epoch: %0.3f" % (pl_model.best_val_dice, pl_model.best_val_dice_epoch), file=f)

        print('-------------------------------------------------------', file=f)

    
if __name__ == "__main__":
    seed = 42
    check_val_every_n_epochs = 1
    max_epochs = 150
    main(seed, check_val_every_n_epochs, max_epochs)