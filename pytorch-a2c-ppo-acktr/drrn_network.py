import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo

import numpy as np

import visualpriors

from config import device, MAP_SIZE, MAP_DIMENSIONS
from model import NNBase, Flatten
from utils import init


import numpy as np
from matplotlib.transforms import Affine2D


class DeepCognitiveMapper(NNBase):
    def __init__(self, num_inputs, mid_level_reps, recurrent=True, hidden_size=2048):
        super(DeepCognitiveMapper, self).__init__(recurrent, hidden_size, hidden_size)

        self.mid_level_reps = mid_level_reps

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.flatten = Flatten()
        
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(8 * len(mid_level_reps)),
            nn.ConvTranspose2d(8 * len(mid_level_reps), 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, padding=(1, 0), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 2, 2, padding=(2, 0), stride=2),
            nn.Sigmoid(),
        )
        

        # For 80x60 input
        self.main = nn.Sequential(
            init_(nn.Conv2d(3, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Print(),
            Flatten(),

            # nn.Dropout(0.2),

            init_(nn.Linear(32 * 7 * 5, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def egomotion_transform(self, previous_map, egomotion):
        
        scale = MAP_SIZE/MAP_DIMENSIONS
        
        x = - egomotion[:,0] * scale
        y = - egomotion[:,2] * scale
        t = - egomotion[:,3]
        
        affine_matrices = []
        for i in range(x.shape[0]):
            if x[i] == 0 and y[i] == 0 and t[i] == 0:
                affine_matrix = np.array([[1, 0, 0],
                                          [0, 1, 0]])
            else:
                affine_matrix = (Affine2D().rotate_around(0,0, t[i]) + Affine2D().translate(
                                tx=x[i], ty=y[i])).get_matrix()
                print(affine_matrix)
            affine_matrices.append(affine_matrix[0:2,:])

            
        T = torch.Tensor(affine_matrices)
        
        grid = F.affine_grid(T, previous_map.size())
        
        update = F.grid_sample(previous_map, grid) 
        
        return update

    def forward(self, inputs, rnn_hxs, masks):
        # print(inputs.size())
        
        inputs = inputs.permute((0, 3, 2, 1))
        x = (inputs[:, 0:3, :, :] / 128.0) - 1

        egomotion = inputs[:, 3, 0, 0:4]

        rep = []
        for feature in self.mid_level_reps:
            rep.append(visualpriors.representation_transform(x, feature, device=device))
        x = torch.concat(rep, 1)  # concatenated mid level representations

        map_update = self.decoder(x)
        previous_map = rnn_hxs.reshape((1,2,32,32))
        previous_map = self.egomotion_transform(previous_map,egomotion)
        
        # new_map = self.combine_maps(previous_map,map_update)

        x = self.main((inputs[:, 0:3, :, :] / 255))
        # # print(x.size())

        # if self.is_recurrent:
        #     x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        rnn_hs = self.flatten(map_update)
        return self.critic_linear(x), x, rnn_hs










































