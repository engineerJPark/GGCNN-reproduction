import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.feature import peak_local_max

def _gr_text_to_no(l, offset=(0,0)):
    '''
    single point from Cornell file string line -> pair of ints
    l : string line
    offset : offset to apply to point pos
    return Point [y, x]
    '''
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]

class GraspRectangles:
    '''
    load and operating on Grasp Rectangle
    '''
    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]
    
    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        '''
        test if GraspRectangle has desired attribute as function and call it
        '''
        if hasattr(GraspRectangles, attr) and callable(GraspRectangles, attr):
            return lambda *args, **kwargs : \
                list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else: 
            raise AttributeError('Couldn\'t find function %s \
                in Bounding Box or Bounding Retangle' % attr)

    @classmethod
    def load_from_array(cls, arr):
        '''
        load grasp rectangles from numpy array
        arr : Nx4x2 array, each 4x2 array is the 4 corner pixels of a grasp rectangle.
        return GraspRectangles() object
        '''
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangles(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_files(cls, fname):
        '''
        load grasp rectangles from a Cornell dataset grasp file.
        fname: Path to text file.
        return GraspRectangles() object
        '''
        grs = []
        with open(fname) as f:
            while True:
                p0 = f.readline()
                if not p0: # EOF
                    break
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])
                    grs.append(GraspRectangles(gr))
                except ValueError:
                    continue
        return cls(grs)

    def append(self, gr):
        pass

    def copy(self):
        pass

    def show(self, ax=None, shape=None):
        pass

    def draq(self, shape, position=True, angle=True, width=True):
        pass

    def to_array(self, pad_to=0):
        pass

    @property
    def center(self):
        pass

class GraspRectangle:
    '''
    representation of a grasp in the common 'Grasp Rectangle format'
    '''
    def __init__(self, ):
        pass
    pass



class Grasp:
    '''
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    
    '''

    pass

def detect_grasps(q_img, ang_img, width_img=None, no_grasp=1):
    '''
    detect grasps in a GGCNN output
    no_grasps: Max number of grasps to return
    return: list of Grasps
    '''
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasp)
    
    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_angle = ang_img[grasp_point]
        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2
        grasps.append(g)
    return grasps