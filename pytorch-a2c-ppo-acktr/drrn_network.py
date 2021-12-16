import torch
import visualpriors
from torch import nn

from config import device
from model import NNBase
from utils import init


class DeepCognitiveMapper(NNBase):
    def __init__(self, num_inputs, mid_level_reps, recurrent=False, hidden_size=128):
        super(DeepCognitiveMapper, self).__init__(recurrent, hidden_size, hidden_size)

        self.mid_level_reps = mid_level_reps

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

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

    def forward(self, inputs, rnn_hxs, masks):
        # print(inputs.size())
        inputs = inputs.permute((0, 3, 1, 2))
        x = (inputs[:, 0:3, :, :] / 128.0) - 1

        egomotion = inputs[:, 3, 0:4, 0]

        rep = []
        for feature in self.mid_level_reps:
            rep.append(visualpriors.representation_transform(x, feature, device=device))
        x = torch.concat(rep, 1)  # concatenated mid level representations

        new_map = self.decoder(x)

        x = self.main((inputs[:, 0:3, :, :] / 255))
        # print(x.size())

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
