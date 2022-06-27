import torch
from skimage.filters import gaussian

def gaussian_filtering(q, cos, sin, width):
    '''
    post process of the ggcnn output by gaussian filter.
    all parameter came from pytorch.

    return filtered Q, Angle, Width.
    '''

    q_post = gaussian(q.cpu().numpy().squeeze(), 2., preserve_range=True)
    ang_post = gaussian((torch.atan2(sin, cos) / 2.).cpu().numpy().squeeze(), 2., preserve_range=True)
    width_post = gaussian(width.cpu().numpy().squeeze() * 150., 2., preserve_range=True)

    return q_post, ang_post, width_post # numpy type