import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAP_SIZE = 2
MAP_DIMENSIONS = 32