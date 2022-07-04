from tkinter import CENTER
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

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        '''
        method to be overide
        get Ground Truth Bounding Box
        '''
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        '''
        method to be overide
        get depth image
        '''
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        '''
        method to be overide
        get rbg image
        '''
        raise NotImplementedError()
    
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
        getting image
        getting grasp 
        '''

        if self.random_rotate:
            rotation = random.choice([0, np.pi/2, 2*np.pi/2, 3*np.pi/2])
        else:
            rotation = 0.
    
        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.
        
        if self.include_depth:
            depth_img = self.get_depth(idx, rotation, zoom_factor)

        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rotation, zoom_factor)

        # load the grasp 
        bbs = self.get_gtbb(idx, rotation, zoom_factor)
        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0., 150.) / 150.

        if self.include_depth and self.include_rgb:
            # make 4 channel image
            # test this
            x = self.numpy_to_torch(
                np.concatenate((np.expand_dims(depth_img, 0), rgb_img), 0)
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.get_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.cos(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rotation, zoom_factor

    def __len__(self):
        '''
        return length of grasp files
        '''
        return len(self.grasp_files)

########################################

class CornellDataset(GraspDatasetBase):
    '''
    Getting CornellDataset
    file_path is used to be '~/catkin_ws/src/cornell'
    '''

    def __init__(self, file_path, start=0., end=1., ds_rotate=0, **kwargs):
        '''
        file_path: Cornell Dataset directory.
        start: If splitting the dataset, start at this fraction [0,1]
        end: If splitting the dataset, finish at this fraction
        ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        kwargs: kwargs for GraspDatasetBase
        '''
        super(CornellDataset, self).__init__(**kwargs)

        # get all the cpos.txt files in the whold dataset
        graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt')) # folders that have grasp file 
        graspf.sort() # dataset fold sorting
        l = len(graspf)

        if l == 0:
            raise FileNotFoundError('No dataset in the file_path')
        
        if ds_rotate:
            graspf = graspf[int(l * ds_rotate):] + graspf[:int(l * ds_rotate)]

        depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
        rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]
        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

    def _get_crop_attrs(self, idx):
        '''
        need to be Identified
        '''
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))

        return center, left, top

    # file download code needed
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        '''
        get Ground Truth Bounding Box
        '''
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        '''
        get depth image
        '''
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        '''
        get rgb image
        '''
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
