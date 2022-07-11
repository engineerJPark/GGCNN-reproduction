from re import X
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
        '''
        Add a grasp rectangle to this GraspRectangles object
        gr: GraspRectangle
        '''
        self.grs.append(gr)

    def copy(self):
        '''
        return: A deep copy of this object and all of its GraspRectangles.
        '''
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        '''
        Draw all GraspRectangles on a matplotlib plot.
        ax: (optional) existing axis
        shape: (optional) Plot shape if no existing axis
        '''
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            ax.imshow(np.zeros[shape])
            ax.axis([0, shape[1], shape[0], 0]) # x축 최소최대, y축 최소최대
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        '''        
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        shape: output shape
        return: Q, Angle, Width outputs (or None)
        '''
        pos_out = np.zeros(shape) if position else None
        ang_out = np.zeros(shape) if angle else None
        width_out = np.zeros(shape) if width else None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            if position: pos_out[rr, cc] = 1.
            if angle: ang_out[rr, cc] = gr.angle
            if width: width_out[rr, cc] = gr.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        '''
        Convert all GraspRectangles to a single array.
        pad_to: Length to 0-pad the array along the first dimension
        return Nx4x2 numpy array
        '''
        a = np.stack([gr.points for gr in self.grs]) # axis 0 = len(self.grs)
        if pad_to and pad_to > len(self.grs):
            a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2)))) # axis = 0
        return a.astype(np.int32)

    @property
    def center(self):
        '''
        Compute mean center of all GraspRectangles
        return: float, mean centre of all GraspRectangles
        '''
        return np.mean(np.vstack([gr.points for gr in self.grs]),\
             axis=0).astype(np.int32)

class GraspRectangle:
    '''
    representation of a grasp in the common 'Grasp Rectangle format'
    '''
    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)


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
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasp) # min distance is too heuristic
    
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