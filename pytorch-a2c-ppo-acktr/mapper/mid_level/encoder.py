import visualpriors
import torch
from torch.utils import model_zoo
from config.config import device


def mid_level_representations(input_image_tensor, representation_names):
    """
    :param input_image_tensor:  (batch_size, 3, 256, 256)
    :param representation_names: list
    :return: concatted image tensor to pass into FCN  (batch_size, 8*len(representation_names), 16, 16)
    """
    representations = []
    for name in representation_names:
        # (batch_size, 3, 256, 256) ——>(batch_size, 8, 16, 16)
        representations.append(visualpriors.representation_transform(input_image_tensor, name, device=device))
    return torch.cat(representations, dim=1)
