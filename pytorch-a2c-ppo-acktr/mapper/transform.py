""" 
mapper/transform.py
----------------------------------------------------------------------------
     Authors : Yongtao Wu, Umer Hasan Sayed, Titouan Renard
     Last Update : Octobre 2021

     Geometric transform functions for out autonomous agent

----------------------------------------------------------------------------
Processing graph:
                                                dX : (3x1) numpy array, (x,y,theta) -- represents robot's motion
                                                |
                                                v
                                         -----------------
    previous_map_transformed  <- ------  |   transform   | <----- previous_map -- (BATCHSIZE x 2 x 256 x 256) tensor
    -> (BATCHSIZE x 2 x 256 x 256)       -----------------
"""

# from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from matplotlib.transforms import Affine2D

from config.config import MAP_SIZE, MAP_DIMENSIONS, device


def egomotion_transform(input_map_tensor, dX):
    """
        Args:
            input_map_tensor: (batch_size, 3, 256, 256)
            dX: change in robots position vector (batch_size, 1, 3)

        Returns: concatenated image tensor to pass into FCN  (batch_size, 8*len(representation_names), 16, 16)
    """

    x = dX[:, :, 0] * (MAP_DIMENSIONS[1] / MAP_SIZE[0])
    y = dX[:, :, 1] * (MAP_DIMENSIONS[2] / MAP_SIZE[1])
    t = dX[:, :, 2]

    affine_transform_vector = -torch.stack((x, y, t), -1)

    return tensor_transform(input_map_tensor, affine_transform_vector)  # call affine transform function


def tensor_transform(input_map_tensor, transform):
    """
    Args:
        input_map_tensor: (batch_size, MAP_DIMENSIONS)
        transform: 

    Returns: concatenated image tensor to pass into FCN  (batch_size, 8*len(representation_names), 16, 16)

    """
    # Construct a 2d rotation and transformation matrix (using scipy functions)
    width, height = MAP_DIMENSIONS[2], MAP_DIMENSIONS[1]

    T = []
    for t in transform:
        t = t[0] # reduce dimension of transform to get actual transform values
        T.append(torch.tensor((Affine2D().rotate_around(width // 2, height // 2, t[2]) + Affine2D().translate(
            tx=t[0], ty=t[1])).get_matrix()[0:2, :], dtype=torch.float, device=device))
    T = torch.stack(T)

    # unsqueezed_tensor = torch.unsqueeze(input_map_tensor,0)
    # (CxHxW) map tensor expanded to a (1xCxHxW) tensor

    # Compute grid transform tensor
    grid = F.affine_grid(T, input_map_tensor.size(),align_corners=False)  # a(1xHxWx2) full affine grid transform tensor
    # Resample input tensor according to grid transform, returns a rotated and translated tensor
    output_map_tensor = F.grid_sample(input_map_tensor, grid, align_corners=False)  # (1xCxHxW) transformed map
    return output_map_tensor
