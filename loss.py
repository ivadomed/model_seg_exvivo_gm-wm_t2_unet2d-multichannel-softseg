import torch
import torch.nn as nn
import scipy
import scipy.ndimage
import numpy as np

class AdapWingLoss(nn.Module):
    """
    Adaptive Wing loss
    Used for heatmap ground truth.

    .. seealso::
        Wang, Xinyao, Liefeng Bo, and Li Fuxin. "Adaptive wing loss for robust face alignment via heatmap regression."
        Proceedings of the IEEE International Conference on Computer Vision. 2019.

    Args:
        theta (float): Threshold between linear and non linear loss.
        alpha (float): Used to adapt loss shape to input shape and make loss smooth at 0 (background).
        It needs to be slightly above 2 to maintain ideal properties.
        omega (float): Multiplicating factor for non linear part of the loss.
        epsilon (float): factor to avoid gradient explosion. It must not be too small
    """
    def __init__(self, theta=0.5, alpha=2.1, omega=14, epsilon=1):
        self.theta = theta
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        super(AdapWingLoss, self).__init__()

    def forward(self, input, target):
        eps = self.epsilon
        # Compute adaptative factor
        A = self.omega * (1 / (1 + torch.pow(self.theta / eps,
                                             self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / eps,
                                              self.alpha - target - 1) * (1 / eps)

        # Constant term to link linear and non linear part
        C = (self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / eps, self.alpha - target)))

        batch_size = target.size()[0]
        hm_num = target.size()[1]

        mask = torch.zeros_like(target)
        kernel = scipy.ndimage.generate_binary_structure(2, 2)
        # For 3D segmentation tasks
        if len(input.shape) == 5:
            kernel = scipy.ndimage.generate_binary_structure(3, 2)

        for i in range(batch_size):
            img_list = list()
            img_list.append(np.round(target[i].cpu().numpy() * 255))
            img_merge = np.concatenate(img_list)
            img_dilate = scipy.ndimage.binary_opening(img_merge, np.expand_dims(kernel, axis=0))
            img_dilate[img_dilate < 51] = 1  # 0*omega+1
            img_dilate[img_dilate >= 51] = 1 + self.omega  # 1*omega+1
            img_dilate = np.array(img_dilate, dtype=int)

            mask[i] = torch.tensor(img_dilate)

        diff_hm = torch.abs(target - input)
        AWingLoss = A * diff_hm - C
        idx = diff_hm < self.theta
        AWingLoss[idx] = self.omega * torch.log(1 + torch.pow(diff_hm / eps, self.alpha - target))[idx]

        AWingLoss *= mask
        sum_loss = torch.sum(AWingLoss)
        all_pixel = torch.sum(mask)
        mean_loss = sum_loss  # / all_pixel

        return mean_loss
