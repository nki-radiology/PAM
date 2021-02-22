import SimpleITK as sitk
import numpy as np

from abc import abstractmethod

from .base import Transform


class Augmentation(Transform):

    def __init__( self ):
        self.random_seed = 0
        super(Augmentation, self).__init__()

    def __call__( self, image, *args ):
        return image

    @abstractmethod
    def get_random_params( self, image ):
        return None


class RandomLinearDisplacement(Augmentation):

    # this code was adapted from keras augmentation module.

    def __init__( self,
                  rotation_range=None,
                  shear_range=None,
                  zoom_range=None,
                  shift_range=None,
                  random_axis_flip=False,
                  interpolator=sitk.sitkLinear,
                  cval=0. ):

        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.random_axis_flip = random_axis_flip
        self.interpolator = interpolator
        self.cval = cval
        super(RandomLinearDisplacement, self).__init__()

    def __call__( self, image, *args ):

        if len(args) == 0:
            args = [self.get_random_params(image)]

        transform_matrix = args[0]

        if len(args) > 1:
            for t in args:
                transform_matrix = transform_matrix.dot(t)

        flt = sitk.AffineTransform(3)
        flt.SetTranslation(tuple(transform_matrix[0:3, -1].squeeze()))
        flt.SetMatrix(transform_matrix[:3, :3].ravel())
        image = sitk.Resample(image, image, flt, self.interpolator, self.cval)
        return image

    def get_random_params( self, image ):

        np.random.seed(self.random_seed)

        transform_matrix = np.eye(4)
        if self.rotation_range is not None:
            R = self._random_rotation()
            transform_matrix = np.dot(transform_matrix, R)
        if self.shear_range is not None:
            S = self._random_shear()
            transform_matrix = np.dot(transform_matrix, S)
        if self.shift_range is not None:
            S = self._random_shift()
            transform_matrix = np.dot(transform_matrix, S)
        if self.zoom_range is not None:
            Z = self._random_zoom()
            transform_matrix = np.dot(transform_matrix, Z)
        if self.random_axis_flip:
            AF = self._random_axis_flip()
            transform_matrix = np.dot(transform_matrix, AF)

        self.random_seed += 1

        return transform_matrix

    def _random_rotation( self ):

        rg = self.rotation_range

        axis_of_rotation = np.random.permutation([0, 1, 2])
        axis_of_rotation = axis_of_rotation[:np.random.randint(low=1, high=4)]

        thetas = [np.pi / 180 * np.random.uniform(-rg, rg) for _ in axis_of_rotation]

        rotation_matrix = np.eye(4)
        for ax, th in zip(axis_of_rotation, thetas):
            c, s = np.cos(th), np.sin(th)
            R = np.eye(4)
            if ax == 0:
                R[:3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            if ax == 1:
                R[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            if ax == 2:
                R[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            rotation_matrix = np.dot(rotation_matrix, R)
        return rotation_matrix

    def _random_shear( self ):

        shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range, 6))

        transform_matrix = np.array([
            [1., shear[0], shear[1], 0.],
            [shear[2], 1., shear[3], 0.],
            [shear[4], shear[5], 1., 0.],
            [0., 0., 0., 1.]
        ])
        return transform_matrix

    def _random_shift( self ):

        if not isinstance(self.shift_range, list):
            self.shift_range = [self.shift_range] * 3

        rg = self.shift_range
        t = [np.random.uniform(-rg[i], rg[i]) for i in range(3)]
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 3] = t
        return transform_matrix

    def _random_zoom( self ):

        if not isinstance(self.zoom_range, list):
            self.zoom_range = [1 + self.zoom_range, 1 - self.zoom_range]

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy, zz = 1, 1, 1
        else:
            zx, zy, zz = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 3)

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = np.diag([zx, zy, zz])

        return transform_matrix

    def _random_axis_flip( self ):

        flip = lambda: np.random.choice([1, -1])
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = np.diag([flip(), flip(), flip()])
        return transform_matrix

    def _transform_matrix_offset_center( self, matrix, x, y, z ):

        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        o_z = float(z) / 2 + 0.5
        offset_matrix, reset_matrix = np.eye(4), np.eye(4)
        offset_matrix[0:3, 3] = [o_x, o_y, o_z]
        reset_matrix[0:3, 3] = [-o_x, -o_y, -o_z]
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix