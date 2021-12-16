import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REPRESENTATION_NAMES = ['keypoints3d', 'depth_euclidean']