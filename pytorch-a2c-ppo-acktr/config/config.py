import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EXPERIMENT_IDS = ['Baseline', 'BaselineMidLevel', 'DRRN', 'DRRNActualMap','DRRNSupervisedMap']
EXPERIMENT_ID_INDEX = 1
CURRENT_POLICY = EXPERIMENT_IDS[EXPERIMENT_ID_INDEX]
REPRESENTATION_NAMES = ['keypoints3d', 'depth_euclidean']