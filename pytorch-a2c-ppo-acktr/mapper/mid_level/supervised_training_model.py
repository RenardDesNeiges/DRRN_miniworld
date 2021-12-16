from torch import nn

from config.config import RESIDUAL_LAYERS_PER_BLOCK, RESIDUAL_NEURON_CHANNEL, RESIDUAL_SIZE, STRIDES, \
    REPRESENTATION_NAMES
from mapper.mid_level.decoder import UpResNet
from mapper.mid_level.fc import FC


class SupervisedTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = FC()
        self.decoder = UpResNet(layers=RESIDUAL_LAYERS_PER_BLOCK, channels=RESIDUAL_NEURON_CHANNEL, sizes=RESIDUAL_SIZE,
                                strides=STRIDES)

    def forward(self, activation):
        # ==========FC==========
        print("Passing fully connected layer...")
        activation = activation.view(activation.shape[0], 1, -1)  # flatten all dimensions except batch,
        # --> tensor of the form (BATCHSIZE x 2048*REPRESENTATION_NUMBER)
        activation = self.fc(activation)  # pass through dense layer --> (BATCHSIZE x 2048*REPRESENTATION_NUMBER) tensor
        activation = activation.view(activation.shape[0], 8 * len(REPRESENTATION_NAMES), 16,
                                     16)  # after fully connected layer, # (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

        # ==========Deconv==========
        print("Passing residual decoder...")
        map_update = self.decoder(activation)  # upsample to map object
        return map_update
