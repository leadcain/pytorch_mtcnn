# -*- coding: utf-8 -*-

import torch
import numpy as np
import torchvision.transforms as transforms
from torch.autograd.variable import Variable

transform = transforms.ToTensor()

#- transform unit8 to [0 ~ 1] fp32
def convert_image_to_tensor(image):
    """convert an image to pytorch tensor

        Parameters:
        ----------
        image: numpy array , h * w * c

        Returns:
        -------
        image_tensor: pytorch.FloatTensor, c * h * w
        """
    # image = image.astype(np.float32)
    return transform(image)


def convert_chwTensor_to_hwcNumpy(tensor):
    """convert a group images pytorch tensor(count * c * h * w) to numpy array images(count * h * w * c)
            Parameters:
            ----------
            tensor: numpy array , count * c * h * w

            Returns:
            -------
            numpy array images: count * h * w * c
            """

    if isinstance(tensor, Variable):
        return np.transpose(tensor.data.numpy(), (0,2,3,1))
    elif isinstance(tensor, torch.FloatTensor):
        return np.transpose(tensor.numpy(), (0,2,3,1))
    else:
        raise Exception("covert b*c*h*w tensor to b*h*w*c numpy error.This tensor must have 4 dimension.")
