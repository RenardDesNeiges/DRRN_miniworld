import torch
import torch.utils.model_zoo

from config.config import REPRESENTATION_NAMES, DEBUG
from mapper.mid_level.encoder import mid_level_representations  # mid_level wrapper class


def encode_with_mid_level(image):
    image = torch.swapaxes(image, 1, 3)
    if DEBUG:
        print(f"Encoding image of shape {image.shape} with mid level encoders.")
    image = mid_level_representations(image, REPRESENTATION_NAMES)
    # (BATCH SIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor
    if DEBUG:
        print(f'Returning encoded representation of shape {image.shape}.')
    return image


def convert_midlevel_to_map(midlevel_representation, fc_network, decoder_network):
    image = midlevel_representation

    if DEBUG:
        print(f"Passing activation of shape {image.shape} through fcn.")

    image = image.view(image.shape[0], 1, -1)
    # flatten all dimensions except batch,
    # --> tensor of the form (BATCHSIZE x 2048*REPRESENTATION_NUMBER)

    image = fc_network(image)
    # pass through dense layer --> (BATCHSIZE x 2048*REPRESENTATION_NUMBER) tensor
    image = image.view(image.shape[0], 8 * len(REPRESENTATION_NAMES), 16, 16)

    # after fully connected layer, #(BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    if DEBUG:
        print(f"Passing activation of shape {image.shape} to decoder.")

    decoder_output = decoder_network(image)

    if DEBUG:
        print(f'Returning output tensor with shape: {decoder_output.shape}')

    return decoder_output
