from curses.textpad import rectangle
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
        '''
        points as gr.points N x 4 x 2
        each 4 points has format as : [position, x/y axis]
        '''
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def as_grasp(self):
        '''
        return GraspRectangle converted to a Grasp
        '''
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        '''
        return: rectangel center point
        '''
        return self.points.mean(axis=0).astype(np.int)
    
    @property
    def length(self):
        '''
        return Rectangle length along the axis of the grasp
        '''
        dx = self.points[1,1] - self.points[0,1]
        dy = self.points[1,0] - self.points[0,0]
        return np.sqrt(dx**2 + dy**2)

    @property
    def width(self):
        '''
        return Rectangel width perpendicular to the axis of the grasp
        '''
        dx = self.points[2,1] - self.points[1,1]
        dy = self.points[2,0] - self.points[1,0]
        return np.sqrt(dx**2 + dy**2)

    def polygon_coords(self, shape=None):
        '''
        shape : output shape
        return indices of pixels, within the grasp rectangle polygon
        '''
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None): 
        '''
        shape : output shape
        return indices of pixels, within the centre third of the grasp rectangle
        '''
        # need to understand
        return Grasp(self.center, self.angle, self.length/3, self.width).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi/6):
        '''
        conpute IoU with another grasping rectangle
        gr : Grasping Rectangel to compare
        angle_threshold : Maximum angle difference bw GraspRectangle
        return IoU between Grasp Rectangles
        '''
        # need to understand
        # self.angle이나 gr.angle이 90도 더 틀어져있나? 
        if abs((self.angle - gr.angle + np.pi/2) % np.pi - np.pi/2) > angle_threshold: 
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try: # for drawing figure, get maximum indices
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas==2)
        return intersection / union

    def copy(self):
        '''
        return copy of itself
        '''
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        '''
        offset grasp rectangle
        offset : array [y, x] distance to offset
        '''
        self.points += np.array(offset).reshape((1,2))

    def rotate(self, angle, center):
        '''
        angle : angle to rotate in radians
        center : point to rotate around like image center
        '''
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)]
            ]
        )
        c = np.array(center).reshape((1,2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int32)

    def scale(self, factor):
        '''
        factor : scale grasp rectangle by factor
        '''
        if factor == 1.:
            return
        self.points *= factor

    def plot(self, ax, color=None):
        '''
        Plot grasping rectangle.
        ax: Existing matplotlib axis
        color: matplotlib color code (optional)
        '''
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        '''
        Zoom grasp rectangle by given factor.
        factor: Zoom factor
        center: Zoom zenter (focus point, e.g. image center)
        '''
        T = np.array(
            [
                [1/factor, 0],
                [0, 1/factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


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