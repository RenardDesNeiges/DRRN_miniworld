""" 
networks/update.py
----------------------------------------------------------------------------
     Authors : Yongtao Wu, Umer Hasan, Titouan Renard
     Last Update : Octobre 2021

     applies the update function to the computed map

----------------------------------------------------------------------------
"""

import torch

from config.config import BATCHSIZE, MAP_DIMENSIONS, device


def update_map(update_matrix, previous_map, eps=1e-6):
    """
    This is the U function defined in our proposal
    :param update_matrix: (batch_size, 2, map_width, map_height)
    :param previous_map: (batch_size, 2, map_width, map_height)
    :param eps: solves divide by 0 problem
    :return: updated_map: (batch_size, 2, map_width, map_height)
    """

    batch_map_dim = (BATCHSIZE, *MAP_DIMENSIONS)

    assert update_matrix.shape == previous_map.shape == batch_map_dim

    updated_map = torch.ones(batch_map_dim).to(device)

    for i in range(BATCHSIZE):
        updated_confidence = (update_matrix[i, 1, :, :] + previous_map[i, 1, :, :] + eps)
        updated_free_space_map = (update_matrix[i, 0, :, :] * update_matrix[i, 1, :, :] + previous_map[i, 0, :, :] * previous_map[i, 1, :, :]) / updated_confidence
        updated_confidence = torch.unsqueeze(updated_confidence, dim=0)
        updated_free_space_map = torch.unsqueeze(updated_free_space_map, dim=0)
        updated_map[i, :, :, :] = torch.cat([updated_free_space_map, updated_confidence], dim=0)

    return updated_map


if __name__ == '__main__':
    a = torch.ones(4, 1, 256, 256) * 2
    b = torch.ones(4, 1, 256, 256) * 3
    c = torch.ones(4, 1, 256, 256) * 4
    d = torch.ones(4, 1, 256, 256) * 5

    update_map(torch.cat([a, b], dim=1), torch.cat([c, d], dim=1))
