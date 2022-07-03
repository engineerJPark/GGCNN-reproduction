import numpy as np
import torch
import torch.utils.data
import random

import os
import glob
from utils.preprocessing import grasp, image

# !pip install kaggle --upgrade
# !kaggle -h
# !kaggle datasets download -d oneoneliu/cornell-grasp
# !unzip '*.zip'

class GraspDatasetBase(torch.utils.data.Dataset):
    # file download code needed
    def __init__(self, output_size, include_depth=True, include_rgb=False,
     random_rotate=False, random_zoom=False, input_only=False):
        '''
        output_size: Image output size in pixels (square)
        include_depth: Whether depth image is included
        include_rgb: Whether RGB image is included
        random_rotate: Whether random rotations are applied
        random_zoom: Whether random zooms are applied
        input_only: Whether to return only the network input (no labels)
        '''
        if include_depth is False and include_rgb is False:
            raise ValueError('at least Depth or RGB image needed')

        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.grasp_files = []
    
    def numpy_to_torch(s): # numpy type s
        '''
        converse to 3 channel image
        '''

        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
    
    def __getitem__(self, idx):
        '''
        rotation and zoom factor
        '''

        if self.random_rotate:
            rotation = random.choice([0, np.pi/2, 2*np.pi/2, 3*np.pi/2])
        else:
            rotation = 0.
    
        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.

        pass


    def __len__(self):
        '''
        return length of grasp files
        '''
        return len(self.grasp_files)

        
            
    


class CornellDataset(GraspDatasetBase):
    # file download code needed
    pass