import pdb

import torch
import torch.nn as nn
from config.config import FC_NEURON_LISTS

# each fc for each representations ?
# use cnn and then fc ?
# (cognitive_mapping_and_planning-master\tfcode\cmp.py # " Add a layer to reduce dimensions for a fc layer.")
class FC(nn.Module):
    """
    3 layer perceptron class nn.Sequential
    output dimensions:
    """
    def __init__(self):
        super().__init__()
        self.neuron_lists=FC_NEURON_LISTS
        fc_layers = []
        for i in range(len(self.neuron_lists)-1):
            fc_layers.append(nn.Linear(self.neuron_lists[i],self.neuron_lists[i+1]))
            # fc_layers.append(nn.BatchNorm1d(self.neuron_lists[i+1])) # TODO don't use batch norm when batchsize = 1
            fc_layers.append(nn.ReLU())
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.fc(x)
        return  x