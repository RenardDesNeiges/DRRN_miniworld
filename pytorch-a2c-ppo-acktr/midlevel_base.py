import torch
import visualpriors
from torch import nn

from config import device
from model import NNBase, Flatten
from utils import init


class MidlevelBase(NNBase):
    def __init__(self, num_inputs, mid_level_reps, recurrent=False, hidden_size=128):
        super(MidlevelBase, self).__init__(recurrent, hidden_size, hidden_size)
        self.mid_level_reps = mid_level_reps
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        num_inputs = 5 * 4 * 8 * len(mid_level_reps)

        # For 80x60 input
        self.main = nn.Sequential(
            Flatten(),
            init_(nn.Linear(num_inputs, 16)),
            nn.ReLU(),
            init_(nn.Linear(16, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        rep = []
        for feature in self.mid_level_reps:
            rep.append(visualpriors.representation_transform(inputs, feature, device=device))
        rep = torch.concat(rep, 1)

        rep = rep / 5  # normalize
        x = self.main(rep)  # (1,128）

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs
