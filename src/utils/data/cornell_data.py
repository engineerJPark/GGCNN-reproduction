import numpy as np
import torch
import torch.utils.data
import random

import os
import glob
from utils.preprocessing import grasp, image

class GraspDatasetBase(torch.utils.data.Dataset):
    # file download code needed
    pass

class CornellDataset(GraspDatasetBase):
    # file download code needed
    pass