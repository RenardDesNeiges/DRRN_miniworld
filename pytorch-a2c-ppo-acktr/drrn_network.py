import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo

import matplotlib.pyplot as plt
import numpy as np

import visualpriors

from config import device, MAP_SIZE, MAP_DIMENSIONS, DEBUG
from model import NNBase, Flatten
from utils import init

import numpy as np
from matplotlib.transforms import Affine2D


class DeepCognitiveMapper(NNBase):
    def __init__(self, num_inputs, mid_level_reps, recurrent=True, hidden_size=128):
        super(DeepCognitiveMapper, self).__init__(recurrent, MAP_DIMENSIONS * MAP_DIMENSIONS * MAP_SIZE, hidden_size)

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
        ).to(device=device)

        # For 80x60 input
        self.encoder = nn.Sequential(
            init_(nn.Conv2d(2, 4, kernel_size=5, stride=1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            init_(nn.Conv2d(4, 4, kernel_size=7, stride=1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            init_(nn.Conv2d(4, 4, kernel_size=7, stride=1)),
            nn.BatchNorm2d(4),
            Flatten(),
            init_(nn.Linear(1024, hidden_size)),
            nn.Tanh()
        ).to(device=device)

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
                                
            affine_matrices.append(affine_matrix[0:2,:])

            
        T = torch.Tensor(affine_matrices).to(device)
        
        grid = F.affine_grid(T, previous_map.size())
        
        update = F.grid_sample(previous_map, grid) 
        
        return update
    
    def combine_maps(self, previous_map, map_update, eps=1e-6):
        updated_map = previous_map

        updated_confidence = (map_update[:, 1, :, :] + previous_map[:, 1, :, :] + eps)
        updated_free_space_map = (map_update[:, 0, :, :] * map_update[:, 1, :, :] + previous_map[:, 0, :, :]\
            * previous_map[:, 1, :, :]) / updated_confidence
        updated_confidence = torch.unsqueeze(updated_confidence, dim=1)
        updated_free_space_map = torch.unsqueeze(updated_free_space_map, dim=1)
        updated_map= torch.cat([updated_free_space_map, updated_confidence], dim=1)
        
        return updated_map

    def forward(self, inputs, rnn_hxs, masks):
        # print(inputs.size())
        
        inputs = inputs.permute((0, 3, 2, 1))
        x = (inputs[:, 0:3, :, :] / 128.0) - 1

        egomotion = inputs[:, 3, 0, 0:4]

        rep = []
        for feature in self.mid_level_reps:
            rep.append(visualpriors.representation_transform(x, feature, device=device))
        x = torch.cat(rep, 1)  # concatenated mid level representations

        map_update = self.decoder(x)
        previous_map = rnn_hxs.reshape((inputs.shape[0],2,32,32))
        previous_map = self.egomotion_transform(previous_map,egomotion)
            
        new_map = self.combine_maps(previous_map,map_update)

        x = self.encoder(new_map)

        if DEBUG:
            fig, (ax1, ax2, ax3,ax4) = plt.subplots(4)
            ax1.imshow(map_update.permute(2,3,1,0)[:,:,1,0])
            ax2.imshow(rnn_hxs.reshape((1,2,32,32)).permute(2,3,1,0)[:,:,1,0])
            ax3.imshow(new_map.permute(2,3,1,0)[:,:,1,0])
            ax4.imshow(egomotion)


        rnn_hs = self.flatten(new_map)
        return self.critic_linear(x), x, rnn_hs










































